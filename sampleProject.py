import os
import torch
from model import GPTConfig, GPT
from sum import sum as my_sum_tool 
import pickle 

def process_tool_call(generated_text):
    """
    Detects [TOOL] expression [/TOOL] and runs the sum.py function.
    """
    if "[TOOL]" in generated_text and "[/TOOL]" in generated_text:
        
        try:
            expression = generated_text.split("[TOOL]")[1].split("[/TOOL]")[0]
            
            result = my_sum_tool(expression) 
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    return None



def evaluate_model(out_dir, test_file):
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load the Model and Meta (for encoding/decoding)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix potential key prefix issues
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load Metadata (stoi/itos)
    meta_path = os.path.join('data', 'tool', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 2. Iterate through your test_problems.txt
    with open(test_file, 'r') as f:
        problems = f.readlines()

    for prob in problems:
        prob = prob.strip()
        if not prob: continue
        
        # Initial Generation
        x = torch.tensor(encode(prob), dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, max_new_tokens=100)[0].tolist()
        output = decode(y)

        # 3. Handle the Tool Loop
        if "[TOOL]" in output and "[/TOOL]" in output:
            expression = output.split("[TOOL]")[1].split("[/TOOL]")[0]
            
            # RUN YOUR SUM.PY FUNCTION
            result = my_sum_tool(expression)
            
            # Feed the answer back to GPT to get the final result
            final_input_str = output + str(result)
            x_final = torch.tensor(encode(final_input_str), dtype=torch.long, device=device)[None, ...]
            y_final = model.generate(x_final, max_new_tokens=20)[0].tolist()
            print(f"Problem: {prob} | Final Output: {decode(y_final)}")
        else:
            print(f"Problem: {prob} | No Tool Call Found | Output: {output}")


if __name__ == "__main__":
    import argparse

    # 1. Setup arguments so you can switch models easily
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out-addition-cot', help='Folder with ckpt.pt')
    parser.add_argument('--test_file', type=str, default='test_problems.txt', help='Your list of math problems')
    args = parser.parse_args()

    # 2. Run the evaluation
    if os.path.exists(args.test_file):
        evaluate_model(args.out_dir, args.test_file)
    else:
        print(f"Error: Could not find {args.test_file}. Please create it first!")
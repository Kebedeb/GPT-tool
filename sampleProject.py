import os
import torch
from model import GPTConfig, GPT
from sum import sum as my_sum_tool 
import wandb
import tiktoken
import re
import pickle

def process_tool_call(generated_text):
    """
    Extracts numbers from [TOOL]...[/TOOL] and calls sum.py.
    """
    # Find only the FIRST [TOOL]...[/TOOL] block
    match = re.search(r"\[TOOL\](.*?)\[/TOOL\]", generated_text)
    if match:
        try:
            expression = match.group(1)
            # Clean out any accidental text the model might have put in the tag
            clean_expr = re.sub(r'[^0-9+]', '', expression)
            
            if '+' in clean_expr:
                parts = clean_expr.split('+')
                return str(my_sum_tool(int(parts[0]), int(parts[1])))
            else:
                # If it's just a single number like [TOOL]9[/TOOL], return it
                return clean_expr
        except:
            return "Error"
    return None
    # match = re.search(r"\[TOOL\](.*?)\[/TOOL\]", generated_text, re.DOTALL)
    
    # if match:
    #     try:
    #         expression = match.group(1) # This is just the "9" or "2+9+0"
    #         # Clean it like we did before
    #         clean_expr = re.sub(r'[^0-9+]', '', expression)
            
    #         # If the model only put one number (like [TOOL]9[/TOOL]),
    #         # we can't split by '+'. Handle that case:
    #         if '+' in clean_expr:
    #             parts = clean_expr.split('+')
    #             return str(my_sum_tool(int(parts[0]), int(parts[1])))
    #         else:
    #             # If it's just a single number, return it as the "result"
    #             return clean_expr
                
    #     except Exception as e:
    #         return f"Error: {e}"
    # return None


    # if "[TOOL]" in generated_text and "[/TOOL]" in generated_text:
        
    #     try:
    #         expression = generated_text.split("[TOOL]")[1].split("[/TOOL]")[0]
    #         clean_expr = re.sub(r'[^0-9+]', '', expression)
        
    #         parts = clean_expr.split('+')
            
    #         if len(parts) == 2:
    #             arg1 = int(parts[0].strip())
    #             arg2 = int(parts[1].strip())
    #             return str(my_sum_tool(arg1, arg2))
    #     except Exception as e:
    #         return f"Error: {e}"
    # return None



def evaluate_model(out_dir, test_file):
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load the Model
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    

    # 2. Setup Tokenizer (TikToken)
    meta_path = os.path.join('data', 'tool', 'meta.pkl')
    if os.path.exists(meta_path):
        print(f"Loading character-level meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("CRITICAL ERROR: meta.pkl not found! Math will be inaccurate.")
        print("Using TikToken encoding (GPT-2)")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # 3. Initialize WandB ONCE (Before the loop)
    wandb.init(
        project="addition-tool-use", 
        name=f"eval-{os.path.basename(out_dir)}",
        config={"test_file": test_file, "model": out_dir})
    
    results_table = wandb.Table(columns=["Problem", "Forced Prompt", "Tool Used", "Result"])

    # 4. Process Problems
    with open(test_file, 'r') as f:
        problems = [line.strip() for line in f if line.strip()]

    for prob in problems:

        ##input_text = prob + "[TOOL]"
        #input_text = prob + " = [TOOL]"
        input_text = prob + "=[TOOL]"
        x = torch.tensor(encode(input_text), dtype=torch.long, device=device)[None, ...]

        
        tokens = model.generate(x, max_new_tokens=60)[0].tolist()
        output = decode(tokens) # Turn numbers into text like "[TOOL]1+2[/TOOL]"
        ##debug check
        print(f"RAW OUTPUT for {prob}: {output}")


        tool_result = process_tool_call(output) # Now the tool call can actually find "[TOOL]"

        if tool_result: 
            
            print(f"🎉 Success! {prob} -> Tool Result: {tool_result}")
        else: 
            
            print(f"❌ Failed: No valid tool call found. Output: {output}")

        results_table.add_data(prob, input_text, tool_result if tool_result else "None",output)
    
    wandb.log({"Evaluation Results": results_table})
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out-addition-cot')
    parser.add_argument('--test_file', type=str, default='test_problems.txt')
    args = parser.parse_args()

    evaluate_model(args.out_dir, args.test_file)

        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
        # prob = prob.strip()
        # if not prob: continue
        
        # --- [HERE IS THE FORCE LOGIC] ---
        # We append [TOOL] so the model HAS to use it.
#         forced_prompt = prob + "[TOOL]"
        
#         print(f"\nProcessing: {forced_prompt}")

#         # Generate
#         x = torch.tensor(encode(forced_prompt), dtype=torch.long, device=device)[None, ...]
#         y = model.generate(x, max_new_tokens=100)[0].tolist()
#         output = decode(y)

#         # 5. Handle the Tool
#         tool_expression = "None"
#         final_answer = "Error"
        
#         if "[TOOL]" in output and "[/TOOL]" in output:
#             try:
#                 # Extract logic (e.g. "2+2")
#                 expression = output.split("[TOOL]")[1].split("[/TOOL]")[0]
#                 tool_expression = expression
                
#                 parts = expression.split('+')
#                 if len(parts) == 2:
#                     arg1 = int(parts[0].strip())            
#                     arg2 = int(parts[1].strip())            
                    
#                     # Run the Tool
#                     result = my_sum_tool(arg1, arg2)
#                     final_answer = str(result)
#                     print(f"🎉 Success! {prob} -> Tool({expression}) -> {final_answer}")
#                 else:
#                     final_answer = "Format Error (No +)"
#             except Exception as e:
#                 final_answer = f"Error: {e}"
#         else:
#             print(f"❌ Failed: Model didn't close the tool tag. Output: {output}")

#         # 6. Log to WandB
#         results_table.add_data(prob, forced_prompt, tool_expression, final_answer)

#     # 7. Finish WandB
#     wandb.log({"Evaluation Results": results_table})
    
#     wandb.finish()
#     # Setup Device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 1. Load the Model and Meta (for encoding/decoding)
#     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     gptconf = GPTConfig(**checkpoint['model_args'])
#     model = GPT(gptconf)
#     state_dict = checkpoint['model']
#     # Fix potential key prefix issues
#     unwanted_prefix = '_orig_mod.'
#     for k,v in list(state_dict.items()):
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     # Load Metadata (stoi/itos)
#     # DELETE OR COMMENT OUT THIS BLOCK
# # if os.path.exists(meta_path):
# #     with open(meta_path, 'rb') as f:
# #         meta = pickle.load(f)
# #     stoi, itos = meta['stoi'], meta['itos']
# #     encode = lambda s: [stoi[c] for c in s]
# #     decode = lambda l: ''.join([itos[i] for i in l])
# # PASTE THIS INSTEAD

#     print("Using TikToken encoding (GPT-2)")
#     enc = tiktoken.get_encoding("gpt2")
#     encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
#     decode = lambda l: enc.decode(l)


#     # 2. Iterate through your test_problems.txt
#     with open(test_file, 'r') as f:
#         problems = f.readlines()

#     for prob in problems:
#         prob = prob.strip()
#         if not prob: continue
        
#         # Initial Generation
#         x = torch.tensor(encode(prob), dtype=torch.long, device=device)[None, ...]
#         y = model.generate(x, max_new_tokens=100)[0].tolist()
#         output = decode(y)

#         # 3. Handle the Tool Loop
#         if "[TOOL]" in output and "[/TOOL]" in output:
#             expression = output.split("[TOOL]")[1].split("[/TOOL]")[0]
#             parts = expression.split('+')
#             if len(parts) == 2:
#                 arg1 = int(parts[0].strip())            
#                 arg2 = int(parts[1].strip())            
#             # RUN YOUR SUM.PY FUNCTION
#                 result = my_sum_tool(arg1,arg2)
#                 return str(result)
#             else:
#                 return "erorr"
            
#             # Feed the answer back to GPT to get the final result
#             final_input_str = output + str(result)
#             x_final = torch.tensor(encode(final_input_str), dtype=torch.long, device=device)[None, ...]
#             y_final = model.generate(x_final, max_new_tokens=20)[0].tolist()
#             print(f"Problem: {prob} | Final Output: {decode(y_final)}")
#         else:
#             print(f"Problem: {prob} | No Tool Call Found | Output: {output}")

#         # 2. Initialize WandB Run
#     wandb.init(
#         project="addition-tool-use", 
#         name=f"eval-{os.path.basename(out_dir)}",
#         config={"test_file": test_file, "model": out_dir}
#     )
    
#     # Create a table to store results
#     results_table = wandb.Table(columns=["Problem", "Model Output", "Tool Used", "Final Answer"])

#     with open(test_file, 'r') as f:
#         problems = f.readlines()

#     for prob in problems:
#         # ... (Keep your existing generation logic) ...
        
#         # 3. Log results to the table
#         results_table.add_data(prob, output, expression if "[TOOL]" in output else "None", decode(y_final))

#     # 4. Finish the run and upload the table
#     wandb.log({"Evaluation Results": results_table})
#     wandb.finish()


# if __name__ == "__main__":
#     import argparse

#     # 1. Setup arguments so you can switch models easily
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--out_dir', type=str, default='out-addition-cot', help='Folder with ckpt.pt')
#     parser.add_argument('--test_file', type=str, default='test_problems.txt', help='Your list of math problems')
#     args = parser.parse_args()

#     # 2. Run the evaluation
#     if os.path.exists(args.test_file):
#         evaluate_model(args.out_dir, args.test_file)
#     else:
#         print(f"Error: Could not find {args.test_file}. Please create it first!")
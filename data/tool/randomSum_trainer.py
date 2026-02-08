import random

def generate_examples(num_examples=5000):
    basic_data = []
    cot_data = []
    scratchpad_data = []

    for _ in range(num_examples):
        # Generate two random numbers (10 to 99)
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        
        a_ones, a_tens = a % 10, a // 10
        b_ones, b_tens = b % 10, b // 10
        
        ones_sum = a_ones + b_ones
        carry = 1 if ones_sum >= 10 else 0
        tens_sum = a_tens + b_tens + carry
        result = a + b

        # 1. BASIC VERSION
        # Syntax: 28+45=[TOOL]8+5[/TOOL]13[TOOL]2+4+1[/TOOL]7 73
        basic = f"{a}+{b}=[TOOL]{a_ones}+{b_ones}[/TOOL]{ones_sum}[TOOL]{a_tens}+{b_tens}+{carry}[/TOOL]{tens_sum} {result}"
        basic_data.append(basic)

        # 2. CHAIN-OF-THOUGHT (CoT)
        # Adds vertical planning steps
        cot = f"{a}+{b}=Let me solve this step by step. Ones: [TOOL]{a_ones}+{b_ones}[/TOOL]{ones_sum}. Tens: [TOOL]{a_tens}+{b_tens}+{carry}[/TOOL]{tens_sum}. Total: {result}"
        cot_data.append(cot)

        # 3. SCRATCHPAD VERSION
        # Explicitly mentions the carry in the text
        scratchpad = f"{a}+{b}=Ones: {a_ones}+{b_ones}={ones_sum}. Scratchpad: carry {carry}. Tens: {a_tens}+{b_tens}+{carry}={tens_sum}. Answer: {result}"
        scratchpad_data.append(scratchpad)

    return basic_data, cot_data, scratchpad_data

# Save to files
basic, cot, scratchpad = generate_examples()

with open('basic_tool_data.txt', 'w') as f:
    f.write('\n'.join(basic))

with open('cot_tool_data.txt', 'w') as f:
    f.write('\n'.join(cot))

with open('scratchpad_tool_data.txt', 'w') as f:
    f.write('\n'.join(scratchpad))

print("Successfully generated 5,000 examples for each version!")
    
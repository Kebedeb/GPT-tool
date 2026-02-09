from sampleProject import process_tool_call

# Test case 1: Standard call
print(process_tool_call("The answer is [TOOL]46+54[/TOOL]")) 
# Should output: '100'

# Test case 2: Bad format
print(process_tool_call("The answer is [TOOL]46-54[/TOOL]")) 
# Should output: 'Format Error (No +)' or similar error
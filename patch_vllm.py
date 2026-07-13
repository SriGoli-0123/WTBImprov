import os
import re

def main():
    target_path = "/scratch/sgoli125/conda/envs/phase1_env/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"
    
    if not os.path.exists(target_path):
        print(f"Error: Target file not found at {target_path}")
        print("Checking default python environment paths...")
        try:
            import vllm
            vllm_path = os.path.dirname(vllm.__file__)
            target_path = os.path.join(vllm_path, "entrypoints/openai/tool_parsers/hermes_tool_parser.py")
        except ImportError:
            print("vLLM not installed in local environment.")
            return

    if not os.path.exists(target_path):
        print(f"Could not find hermes_tool_parser.py. Please verify path: {target_path}")
        return

    print(f"Found hermes_tool_parser.py at: {target_path}")
    
    with open(target_path, "r") as f:
        content = f.read()

    # Find the target block using flexible whitespace regex to get exact indentation and variable name
    pattern = r"^([ \t]*)raw_function_calls\s*=\s*\[\s*json\.loads\(\s*match\[0\]\s*if\s*match\[0\]\s*else\s*match\[1\]\s*\)\s*for\s*match\s*in\s*([a-zA-Z0-9_]+)\s*\]"
    
    match = re.search(pattern, content, re.MULTILINE)
    
    if match:
        indent = match.group(1)
        matches_var = match.group(2)
        # Construct replacement string by concatenation to avoid backslashes inside f-strings (for python < 3.12 compatibility)
        # Prepend indent to raw_function_calls so it aligns correctly inside the try block
        replacement = (
            indent + "raw_function_calls = []\n"
            + indent + "for match in " + matches_var + ":\n"
            + indent + "    s = match[0] if match[0] else match[1]\n"
            + indent + "    # Robustly extract JSON block using regex\n"
            + indent + "    json_match = re.search(r\"\\{.*\\}\", s, re.DOTALL)\n"
            + indent + "    if json_match:\n"
            + indent + "        s = json_match.group(0)\n"
            + indent + "    try:\n"
            + indent + "        raw_function_calls.append(json.loads(s))\n"
            + indent + "    except Exception as e:\n"
            + indent + "        print(f\"[vLLM Patch Warning] Failed to parse tool call: {s}. Error: {e}\", flush=True)"
        )
        
        # Replace the matched block with the new implementation
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(target_path, "w") as f:
            f.write(new_content)
        print("Successfully patched hermes_tool_parser.py using regex alignment!")
        
    else:
        # Check if already patched
        if "[vLLM Patch Warning]" in content:
            print("File is already patched.")
        else:
            print("Error: Could not locate the target json.loads block in the file.")
            print("Please check file content around lines 130-150.")

if __name__ == "__main__":
    main()

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

    # Step 1: Self-healing check. Revert any corrupted/previous patch attempt if present.
    if "raw_decode" in content or "[vLLM Patch Warning]" in content:
        print("Detecting previous patch. Reverting to original state...")
        # Regex matches our entire patched loop structure and isolates the original matches variable
        corrupted_pattern = r"(?:[ \t]*)raw_function_calls\s*=\s*\[\]\s*\n\s*for match in ([a-zA-Z0-9_]+):.*?(?:flush=True\)|pos \+= 1\s*\n\s*tool_calls)"
        
        # Original clean block replacement
        original_block = (
            "                raw_function_calls = [\n"
            "                    json.loads(match[0] if match[0] else match[1])\n"
            "                    for match in \\1\n"
            "                ]"
        )
        
        content = re.sub(corrupted_pattern, original_block, content, flags=re.DOTALL)
        
        with open(target_path, "w") as f:
            f.write(content)
        print("Successfully reverted previous patch. Proceeding to apply updated patch...")

    # Step 2: Apply the correct regex aligned patch with dictionary type-checks
    pattern = r"^([ \t]*)raw_function_calls\s*=\s*\[\s*json\.loads\(\s*match\[0\]\s*if\s*match\[0\]\s*else\s*match\[1\]\s*\)\s*for\s*match\s*in\s*([a-zA-Z0-9_]+)\s*\]"
    
    match = re.search(pattern, content, re.MULTILINE)
    
    if match:
        indent = match.group(1)
        matches_var = match.group(2)
        
        # Construct replacement block where we strictly verify raw_decode output is a dictionary representation
        # of a tool call (preventing raw string parses from causing TypeError: string indices must be integers)
        replacement = (
            indent + "raw_function_calls = []\n"
            + indent + "for match in " + matches_var + ":\n"
            + indent + "    s = match[0] if match[0] else match[1]\n"
            + indent + "    decoder = json.JSONDecoder()\n"
            + indent + "    pos = 0\n"
            + indent + "    parsed_any = False\n"
            + indent + "    while pos < len(s):\n"
            + indent + "        while pos < len(s) and s[pos] in chr(32) + chr(9) + chr(10) + chr(13) + \",\":\n"
            + indent + "            pos += 1\n"
            + indent + "        if pos >= len(s):\n"
            + indent + "            break\n"
            + indent + "        try:\n"
            + indent + "            obj, new_pos = decoder.raw_decode(s, pos)\n"
            + indent + "            if isinstance(obj, dict) and \"name\" in obj:\n"
            + indent + "                raw_function_calls.append(obj)\n"
            + indent + "                parsed_any = True\n"
            + indent + "            pos = new_pos\n"
            + indent + "        except Exception:\n"
            + indent + "            pos += 1\n"
            + indent + "    if not parsed_any:\n"
            + indent + "        print(f\"[vLLM Patch Warning] Failed to parse tool call from: {s}\", flush=True)"
        )
        
        # Replace the matched block with the new implementation
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        with open(target_path, "w") as f:
            f.write(new_content)
        print("Successfully patched hermes_tool_parser.py using regex alignment!")
        
    else:
        # Check if already patched
        if "raw_decode" in content:
            print("File is already patched with raw_decode parser.")
        else:
            print("Error: Could not locate the target json.loads block in the file.")

if __name__ == "__main__":
    main()

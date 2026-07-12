import os
import re

def main():
    target_path = "/scratch/sgoli125/conda/envs/phase1_env/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"
    
    if not os.path.exists(target_path):
        print(f"Error: Target file not found at {target_path}")
        print("Checking default python environment paths...")
        # Fallback: try to find vllm in current python site-packages
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

    # Define target snippet to replace
    target_pattern = """        raw_function_calls = [
            json.loads(match[0] if match[0] else match[1])
            for match in matches
        ]"""

    # Replacement snippet with robust JSON extraction and try-except safety
    replacement = """        raw_function_calls = []
        for match in matches:
            s = match[0] if match[0] else match[1]
            # Robustly extract JSON block using regex to avoid trailing text/comments
            json_match = re.search(r"\\{.*\\}", s, re.DOTALL)
            if json_match:
                s = json_match.group(0)
            try:
                raw_function_calls.append(json.loads(s))
            except Exception as e:
                # Print warning and skip invalid candidate instead of crashing server
                print(f"[vLLM Patch Warning] Failed to parse tool call: {s}. Error: {e}", flush=True)"""

    if target_pattern in content:
        new_content = content.replace(target_pattern, replacement)
        with open(target_path, "w") as f:
            f.write(new_content)
        print("Successfully patched hermes_tool_parser.py!")
    else:
        # Check if already patched
        if "[vLLM Patch Warning]" in content:
            print("File is already patched.")
        else:
            print("Error: Could not locate the target json.loads block in the file. Check if file version matches.")

if __name__ == "__main__":
    main()

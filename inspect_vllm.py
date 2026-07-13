import os

def main():
    target_path = "/scratch/sgoli125/conda/envs/phase1_env/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"
    
    if not os.path.exists(target_path):
        print(f"Error: Target file not found at {target_path}")
        return

    with open(target_path, "r") as f:
        lines = f.readlines()

    # Find where extract_tool_calls is defined
    start_line = -1
    for i, line in enumerate(lines):
        if "def extract_tool_calls" in line:
            start_line = i
            break

    if start_line == -1:
        print("Could not find extract_tool_calls in the file.")
        return

    print(f"--- Printing lines from {start_line} to {start_line + 40} ---")
    for idx in range(start_line, min(start_line + 40, len(lines))):
        print(f"{idx+1:03d}: {lines[idx]}", end="")

if __name__ == "__main__":
    main()

import os
import re
import sys

def main():
    target_path = "/scratch/sgoli125/conda/envs/phase1_env/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"

    # Optional explicit path: python3 patch_vllm.py /path/to/hermes_tool_parser.py
    if len(sys.argv) > 1:
        target_path = sys.argv[1]

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
        # Regex matches our entire patched loop structure and isolates the original
        # matches variable. Anchored on the final "[vLLM Patch Warning]" print so it
        # spans the whole block even when the block contains earlier flush=True prints.
        corrupted_pattern = r"(?:[ \t]*)raw_function_calls\s*=\s*\[\]\s*\n\s*for match in ([a-zA-Z0-9_]+):(?:.*?\[vLLM Patch Warning\].*?flush=True\)|.*?pos \+= 1\s*\n\s*tool_calls)"
        
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
        
        # Construct replacement block:
        # 1) iterative raw_decode extracts one or more complete tool-call dicts;
        # 2) if none parsed, repair truncated JSON (the common temp>0 failure:
        #    generation stops before closing braces) by balancing quotes and
        #    brackets; if the cut landed mid-token (e.g. '..., {"t'), progressively
        #    trim back to the last complete element and retry - so the call is
        #    preserved instead of dropped;
        # 3) only warn when even the trimmed-and-repaired string is unusable.
        replacement = (
            indent + "raw_function_calls = []\n"
            + indent + "for match in " + matches_var + ":\n"
            + indent + "    s = match[0] if match[0] else match[1]\n"
            + indent + "    def _wtb_balance(t):\n"
            + indent + "        stack = []\n"
            + indent + "        in_str = False\n"
            + indent + "        esc = False\n"
            + indent + "        for ch in t:\n"
            + indent + "            if esc:\n"
            + indent + "                esc = False\n"
            + indent + "                continue\n"
            + indent + "            if ch == chr(92) and in_str:\n"
            + indent + "                esc = True\n"
            + indent + "                continue\n"
            + indent + "            if ch == chr(34):\n"
            + indent + "                in_str = not in_str\n"
            + indent + "                continue\n"
            + indent + "            if in_str:\n"
            + indent + "                continue\n"
            + indent + "            if ch in \"{[\":\n"
            + indent + "                stack.append(ch)\n"
            + indent + "            elif ch == \"}\" and stack and stack[-1] == \"{\":\n"
            + indent + "                stack.pop()\n"
            + indent + "            elif ch == \"]\" and stack and stack[-1] == \"[\":\n"
            + indent + "                stack.pop()\n"
            + indent + "        out = t\n"
            + indent + "        if in_str:\n"
            + indent + "            out += chr(34)\n"
            + indent + "        for opener in reversed(stack):\n"
            + indent + "            out += \"}\" if opener == \"{\" else \"]\"\n"
            + indent + "        return out\n"
            + indent + "    def _wtb_fix(o):\n"
            + indent + "        a = o.get(\"arguments\")\n"
            + indent + "        if isinstance(a, str):\n"
            + indent + "            try:\n"
            + indent + "                a = json.loads(a)\n"
            + indent + "            except Exception:\n"
            + indent + "                a = {}\n"
            + indent + "        if not isinstance(a, dict):\n"
            + indent + "            a = {}\n"
            + indent + "        o[\"arguments\"] = a\n"
            + indent + "        return o\n"
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
            + indent + "                obj.setdefault(\"arguments\", {})\n"
            + indent + "                raw_function_calls.append(_wtb_fix(obj))\n"
            + indent + "                parsed_any = True\n"
            + indent + "            pos = new_pos\n"
            + indent + "        except Exception:\n"
            + indent + "            pos += 1\n"
            + indent + "    if not parsed_any:\n"
            + indent + "        start = s.find(\"{\")\n"
            + indent + "        if start != -1:\n"
            + indent + "            work = s[start:]\n"
            + indent + "            tries = 0\n"
            + indent + "            while work and tries < 60 and not parsed_any:\n"
            + indent + "                try:\n"
            + indent + "                    obj = json.loads(_wtb_balance(work))\n"
            + indent + "                    if isinstance(obj, dict) and \"name\" in obj:\n"
            + indent + "                        obj.setdefault(\"arguments\", {})\n"
            + indent + "                        raw_function_calls.append(_wtb_fix(obj))\n"
            + indent + "                        parsed_any = True\n"
            + indent + "                        print(\"[vLLM Patch Info] Repaired truncated tool call: \" + str(obj.get(\"name\")), flush=True)\n"
            + indent + "                        break\n"
            + indent + "                except Exception:\n"
            + indent + "                    pass\n"
            + indent + "                cut = max(work.rfind(\",\"), work.rfind(\"{\"), work.rfind(\"[\"), work.rfind(chr(34)))\n"
            + indent + "                if cut <= 0:\n"
            + indent + "                    break\n"
            + indent + "                work = work[:cut].rstrip().rstrip(\",\")\n"
            + indent + "                tries += 1\n"
            + indent + "    if not parsed_any:\n"
            + indent + "        print(\"[vLLM Patch Warning] Dropped unparseable tool call: \" + s[:200], flush=True)"
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

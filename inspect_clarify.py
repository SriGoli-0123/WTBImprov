import json
from pathlib import Path
import re

# Paths
base_dir = Path(__file__).parent.resolve()
result_file = base_dir / "WildToolBench" / "wild-tool-bench" / "result" / "Qwen_Qwen3-14B" / "Wild-Tool-Bench_result_raw.jsonl"

if not result_file.exists():
    result_file = base_dir / "WildToolBench" / "wild-tool-bench" / "result" / "Qwen_Qwen3-14B" / "Wild-Tool-Bench_result.jsonl"

print(f"Scanning {result_file} for other tool call formats...")

unparsed_xml = 0
unparsed_json = 0
other_formats = []

with open(result_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        results = entry.get("result", [])
        for turn in results:
            inf_log = turn.get("inference_log", {})
            for step_key, step_val in inf_log.items():
                if not step_key.startswith("step_"):
                    continue
                output = step_val.get("inference_output", {})
                content = output.get("content", "")
                tool_calls = output.get("tool_calls", [])
                
                # We only care about steps that failed (current_action_name_label == error)
                # and had empty tool_calls in the raw file
                if output.get("current_action_name_label") == "error" and (not tool_calls or len(tool_calls) == 0):
                    # Check for any xml-like tags or json-like structures
                    if "<tool_call>" in content:
                        # This is the standard one we already parse, but let's check if it failed parsing
                        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
                        if match:
                            tool_call_str = match.group(1).strip()
                            try:
                                json.loads(tool_call_str)
                            except:
                                unparsed_json += 1
                                if len(other_formats) < 5:
                                    other_formats.append(("unparsed_json", content))
                    else:
                        # Check for other tags like <call>, <function>, etc.
                        other_tags = re.findall(r"<([a-zA-Z0-9_/]+)>", content)
                        if other_tags:
                            # print the content if it looks like a tool call
                            if any(x in content.lower() for x in ["call", "function", "tool"]):
                                unparsed_xml += 1
                                if len(other_formats) < 5:
                                    other_formats.append(("other_tag", content))

print(f"Unparsed XML/tag structures: {unparsed_xml}")
print(f"XML tool calls with unparsed JSON inside: {unparsed_json}")
print()
print("Examples of other formats:")
for fmt_type, content in other_formats:
    print("-" * 50)
    print(f"Type: {fmt_type}")
    print(repr(content[:500]) + "...")

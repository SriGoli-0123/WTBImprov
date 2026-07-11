import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert WildToolBench JSON metrics to a formatted Markdown report.")
    parser.add_argument("--model", type=str, default="Qwen_Qwen2.5-7B-Instruct", help="Model folder name (e.g. Qwen_Qwen2.5-7B-Instruct)")
    args = parser.parse_args()
    
    metrics_path = Path(f"score/{args.model}/Wild-Tool-Bench_metric.json")
    output_path = Path(f"score/{args.model}/Wild-Tool-Bench_metric.md")
    
    if not metrics_path.exists():
        print(f"[-] Error: Metrics file not found at {metrics_path}")
        print(f"    Search path tried: {metrics_path.resolve()}")
        return
        
    with open(metrics_path) as f:
        data = json.load(f)
        
    model_name = data.get("model_name", args.model)
    
    md_content = []
    md_content.append(f"# 📊 WildToolBench Performance Report: `{model_name}`\n")
    
    # 1. Total Info
    md_content.append("## 📈 Overall Accuracy")
    md_content.append("| Metric Level | Accuracy | Correct Count | Total Count |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for level, stats in data.get("total_info", {}).items():
        acc = stats.get("accuracy", 0.0) * 100
        md_content.append(f"| **{level.capitalize()}** | **{acc:.2f}%** | {stats.get('correct_count')} | {stats.get('total_count')} |")
    md_content.append("\n")
    
    # 2. Task Type Info
    md_content.append("## 🛠️ Performance by Task Type")
    md_content.append("| Task Type | Accuracy | Correct Count | Total Count |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for t_type, stats in data.get("task_type_info", {}).items():
        acc = stats.get("accuracy", 0.0) * 100
        md_content.append(f"| {t_type} | {acc:.2f}% | {stats.get('correct_count')} | {stats.get('total_count')} |")
    md_content.append("\n")
    
    # 3. Layer Info
    md_content.append("## 🥞 Performance by Layer (Dialogue Depth)")
    md_content.append("| Turn (Layer) | Accuracy | Correct Count | Total Count |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for layer, stats in data.get("layer_info", {}).items():
        acc = stats.get("accuracy", 0.0) * 100
        md_content.append(f"| Turn {layer} | {acc:.2f}% | {stats.get('correct_count')} | {stats.get('total_count')} |")
    md_content.append("\n")
    
    # 4. Turn Subtype Info
    md_content.append("## 🧠 Performance by Turn Subtype (Cognitive Patterns)")
    md_content.append("| Turn Subtype | Accuracy | Correct Count | Total Count |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for subtype, stats in data.get("turn_subtype_info", {}).items():
        acc = stats.get("accuracy", 0.0) * 100
        md_content.append(f"| {subtype} | {acc:.2f}% | {stats.get('correct_count')} | {stats.get('total_count')} |")
    md_content.append("\n")
    
    # 5. Optimal Info
    md_content.append("## ⏱️ Optimal Rate (Orchestration Efficiency)")
    md_content.append("| Category | Optimal Path Accuracy | Correct Count | Total Count |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for cat, stats in data.get("optimal_info", {}).items():
        acc = stats.get("accuracy", 0.0) * 100
        md_content.append(f"| {cat} | {acc:.2f}% | {stats.get('correct_count')} | {stats.get('total_count')} |")
    md_content.append("\n")

    # 6. Progress Info
    md_content.append("## 📈 Chaining Execution Progress (Step Completion)")
    md_content.append("| Category | Step Completion Rate | Completed Steps | Total Expected Steps |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for cat, stats in data.get("progress_info", {}).items():
        rate = stats.get("rate", 0.0) * 100
        md_content.append(f"| {cat} | {rate:.2f}% | {stats.get('complete_step')} | {stats.get('total_step')} |")
    md_content.append("\n")

    with open(output_path, "w") as fout:
        fout.write("\n".join(md_content))
    print(f"[+] Saved beautiful Markdown metrics report to: {output_path.resolve()}")

if __name__ == "__main__":
    main()

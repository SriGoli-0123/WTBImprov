# IGAR-v24 restored run

This branch keeps every later experiment and restores the complete IGAR-v24
runtime used by the historical run. Nothing else needs to be selected: when no
`WTB_METHOD` override is set, Qwen uses the stock OpenAI-compatible handler and
the historical IGAR-v24 logic in `BaseHandler`.

The restored files are byte-for-byte copies of the historical versions:

- `wtb/model_handler/base_handler.py` from `ff3d4a1`
- `wtb/model_handler/api_inference/oai.py` from `ff3d4a1`
- repository-root `patch_vllm.py` from `ff3d4a1`

The saved historical result was 404/1024 correct tasks and 12/256 complete
sessions. A rerun must use a new result directory because the generator resumes
and skips entries already present in an old directory.

## 1. Pull and verify

Run from the repository root:

```bash
cd ~/wildtoolbench_workspace_vllm
git switch demo2
git pull --ff-only origin demo2
git branch --show-current
git log -1 --oneline

git hash-object WildToolBench/wild-tool-bench/wtb/model_handler/base_handler.py
git hash-object WildToolBench/wild-tool-bench/wtb/model_handler/api_inference/oai.py
git hash-object patch_vllm.py
```

The three hashes must be:

```text
29e47107f8a0cca066e04ea161e9f6d536f34adc
5a8ae6b30d6bacd67950365cd3246bf855c390a2
63690b8195bef0fef2b7c5b040f0a904fca2a824
```

## 2. Patch and start vLLM

Stop the currently running vLLM server before this step. Apply the historical
Hermes parser patch, then start a fresh server so the patched parser is loaded:

```bash
cd ~/wildtoolbench_workspace_vllm
python3 patch_vllm.py

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Leave that terminal running.

## 3. Run IGAR-v24 in a second terminal

These commands deliberately do not pass a temperature argument. They use four
workers, matching the command recorded with the historical setup.

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
conda activate phase1_env

unset WTB_METHOD WTB_SC_N WTB_ENTITY_LABELS WTB_ASK_GATE

python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 4 \
  --result-dir result_igar_v24_restored

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_igar_v24_restored \
  --score-dir score_igar_v24_restored
```

Do not prefix the evaluation command with `WTB_METHOD=prism`, `concord`,
`gavel`, or `grounded`. The unprefixed command is the simple IGAR-v24 path.

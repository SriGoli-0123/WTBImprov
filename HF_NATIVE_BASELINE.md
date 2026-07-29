# Native Hugging Face Qwen Baseline

This branch reproduces the WildToolBench open-source inference setup without
vLLM, an OpenAI-compatible server, or a server-side tool parser.

For the exact model name `Qwen/Qwen2.5-7B-Instruct`, WTB now:

1. loads the tokenizer and model directly with Transformers;
2. formats the original date-only system message, visible conversation, and
   supplied tools using Qwen's native tokenizer chat template;
3. calls `model.generate` with the default generation configuration and only
   `max_new_tokens=512`;
4. parses complete native `<tool_call>...</tool_call>` blocks without repairing
   or inventing calls.

No WTB answer, task type, score, or tool-call graph is passed to the model
handler. The outer benchmark loop remains unchanged.

## Installation

From `WildToolBench/wild-tool-bench`:

```bash
python3 -m pip install -r ../requirements.txt
python3 -m pip install -r requirements-hf.txt
```

The cluster environment must already provide a CUDA-compatible PyTorch build.
For strict paper-version comparison, confirm both CUDA and Transformers:

```bash
python3 -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

CUDA should print `True`; the expected Transformers version is `4.51.0`.

## Full run

Do not start vLLM. Use one WTB worker because one directly loaded model is
shared in-process:

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --temperature 0 \
  --num-threads 1 \
  --result-dir result_hf_native_baseline
```

Then score the fresh result:

```bash
python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_hf_native_baseline \
  --score-dir score_hf_native_baseline
```

## Optional environment variables

- `WTB_HF_MODEL_PATH`: an already-downloaded local model directory;
- `WTB_HF_REVISION`: a Hugging Face model revision;
- `WTB_HF_LOCAL_FILES_ONLY=1`: forbid downloads and require cached files;
- `WTB_HF_DTYPE`: `auto` (default), `bfloat16`, `float16`, or `float32`;
- `WTB_HF_DEVICE_MAP`: Transformers device map, default `auto`.

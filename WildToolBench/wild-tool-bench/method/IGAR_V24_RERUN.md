# Exact IGAR-v24 rerun

This branch restores `wtb/model_handler/base_handler.py` and
`wtb/model_handler/api_inference/oai.py` byte-for-byte from commit
`ff3d4a19a648c346ed773f44f5b6dd9d8a5ef52f`, the complete runtime whose
recorded result was added in commit `2f1e55d1620c3ee6751f5bb7663b4c67527e9bd5`.

The historical recorded Qwen2.5-7B score was 404/1024 correct tasks and 12/256
complete sessions. A new run is needed to determine whether that result is
reproducible under the current inference server and software environment.

IGAR-v24 modifies the common `BaseHandler`; `WTB_METHOD=igar_v24` deliberately
selects the stock OpenAI-compatible handler. Do not use `WTB_METHOD=prism`,
`concord`, `gavel`, or `grounded` for this reproduction.

## Verify the exact version

```bash
git switch demo2
git pull --ff-only origin demo2

git hash-object wtb/model_handler/base_handler.py
git rev-parse ff3d4a1:WildToolBench/wild-tool-bench/wtb/model_handler/base_handler.py

git hash-object wtb/model_handler/api_inference/oai.py
git rev-parse ff3d4a1:WildToolBench/wild-tool-bench/wtb/model_handler/api_inference/oai.py

WTB_METHOD=igar_v24 python3 -B -c "
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"
```

Each local hash must match the historical hash below it, and the handler must
print `OpenAIHandler`.

## Run into fresh directories

```bash
WTB_METHOD=igar_v24 python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --temperature 0.0 \
  --num-threads 8 \
  --result-dir result_igar_v24_rerun

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_igar_v24_rerun \
  --score-dir score_igar_v24_rerun
```

Use the defaults `WTB_SC_N=1`, `WTB_ENTITY_LABELS=0`, and `WTB_ASK_GATE=1`.
Unset any shell overrides for those variables before running if they were
changed for an earlier experiment.

## Important boundary

This is a historical reproducibility arm, not the evaluator-free method.
IGAR-v24 constructs extra ledger, surfaced-fact, and dialogue-state system
messages from the benchmark-provided history. That is why it is being restored
only to check whether the reported 12-session result repeats.

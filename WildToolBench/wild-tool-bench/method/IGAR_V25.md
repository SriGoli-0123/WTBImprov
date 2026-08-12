# IGAR-v25: deterministic authority separation

IGAR-v25 is an additive, conservative correction to the restored IGAR-v24
runtime. The v24 path remains available and unchanged.

The repeated v24 comparison exposed two general problems:

1. Python set ordering changed the order of required slots in 151 model inputs,
   despite the underlying session information being identical.
2. Values invented in an assistant clarification could appear in the following
   call and pass v24's grounding check merely because the assistant had already
   written them.

V25 makes required-slot state follow the stable order in each tool schema. It
also separates information by authority:

- trusted: system records, user messages, and tool observations;
- untrusted for argument grounding: assistant-authored prose and draft values.

Assistant tool-call history is still retained for detecting repeated calls.
V25 adds no training, RL, or new behavioral system instruction. It also does
not inspect evaluation scores or current-turn labels. However, it deliberately
inherits v24's reference-history ledger, so this is a controlled v24 upgrade,
not yet the final benchmark-independent replacement for that ledger.

## Run

Start the patched vLLM server exactly as for the restored v24 run. Then use a
fresh result directory:

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
conda activate phase1_env

unset WTB_SC_N WTB_SC_TEMP WTB_ENTITY_LABELS WTB_ASK_GATE
export WTB_METHOD=igar_v25

python3 -B -c "
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"

python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 4 \
  --result-dir result_igar_v25

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_igar_v25 \
  --score-dir score_igar_v25
```

The verification command must print `IGARV25Handler`.

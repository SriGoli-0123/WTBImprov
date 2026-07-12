# Consensus Decoding + Action-Mode Triage for WildToolBench

Inference-time method to raise task- and session-level accuracy of small open models
(e.g. Qwen2.5-7B-Instruct) on WildToolBench, replacing the three earlier
"pre-execution check" prompt methods.

## Why the previous methods could not work

Failure analysis of the baseline (`score/Qwen_Qwen2.5-7B-Instruct`, task acc 34.4%,
session acc 1.6%) shows **416 of 672 failures are action-name errors** — the model
chose the wrong *kind* of response — and only 256 are argument errors:

| Failure pattern | Count | Meaning |
|---|---|---|
| should ASK, called tools instead | 144 | guessed/hallucinated missing required params |
| should call TOOLS, replied with text | 93 | narrated intent or asked unnecessarily |
| should ANSWER from history, called tools | 81 | re-queried data already in the conversation |
| wrong / partially wrong tool set | 98 | tool selection or grouping errors |
| argument key-set mismatch | 148 | added unrequested optional params (`sync: true`, `last_knowledge_of_server: 0`, …) |
| argument value errors | ~108 | wrong dates/IDs/strings |

All three earlier methods rewrote **arguments of an already-emitted tool call**.
They ran *after* the action-mode decision, so they could not fix any of the 416
action-name errors, while their out-of-band rewriting corrupted arguments that were
already correct (task accuracy dropped to 25.3% / 33.9% / 23.6%). Session accuracy
(all 4 tasks of a session correct) is roughly `task_acc^4`, so nothing short of a
broad task-accuracy lift can move it.

## The method: how a careful human avoids these mistakes

Two orthogonal, benchmark-agnostic components, both in
`WildToolBench/wild-tool-bench/wtb/model_handler/`:

### 1. Action-mode triage protocol (`base_handler.py`, `SYSTEM_PROMPT_TEMPLATE`)

*A human decides **what kind** of response the situation needs before acting.*
The baseline system prompt was only `Current Date: ...` — the model got zero
behavioral guidance. The new system prompt makes the model classify every turn into
one of three modes before responding:

- **[A] CALL TOOLS** — only when every required parameter is actually available;
- **[B] ASK** — when a required parameter is missing or the reference is ambiguous
  ("one of them"); never guess, never use placeholders;
- **[C] ANSWER** — when the conversation (incl. earlier tool results) already
  contains the answer; never re-call a tool for known data.

plus five tool-calling rules: all independent calls in one turn (parallel),
verbatim value copying, relative-date resolution against Current Date, argument
minimalism (no self-invented optional params — directly targets the 148 key-set
mismatches), and "never announce a tool call in text — make it".

### 2. Consensus decoding / step-level self-consistency (`base_handler.py` + `api_inference/oai.py`)

*A human double-checks by re-deriving the answer independently and going with the
consensus.* At every step the handler samples `WTB_SC_N` candidates — 1 **anchor**
at the run temperature plus N−1 diversity samples at `WTB_SC_TEMP` (one batched
vLLM request via the OpenAI `n` parameter, so the prompt is prefilled once) — and
votes hierarchically:

1. **Mode + tool-name multiset**: candidates cluster by signature
   (`text` vs sorted tool names). Largest cluster wins; ties go to the anchor's
   cluster. Random flip-flops between call/ask/answer and one-off wrong-tool picks
   get outvoted.
2. **Argument keys**: within the winning cluster, a parameter is kept only if a
   strict majority of candidates include it (schema-required params are always
   kept). Hallucinated optional params rarely survive a majority.
3. **Argument values**: each kept key takes its majority value (canonical-JSON
   vote, ties resolved toward the anchor).

Sampling failures degrade gracefully to the anchor-only response, and handlers
without batched sampling (DeepSeek/HunYuan) automatically run anchor-only.
The vote for each step is recorded in the result JSONL under
`inference_log.step_k.consensus` for later analysis.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `WTB_SC_N` | 5 | candidates per step; `1` disables voting entirely |
| `WTB_SC_TEMP` | 0.8 | temperature of the diversity samples |

## Running on SOL

Same command as before:

```bash
sbatch run_wtb_vllm.sh Qwen/Qwen2.5-7B-Instruct
```

Notes:
- Results are written to **`result_consensus/` + `score_consensus/`** so the
  committed baseline in `result/` + `score/` stays intact for comparison. (The
  generator skips ids that already have results, so re-using `result/` would
  silently skip the whole run.)
- The script now passes `--enable-auto-tool-choice --tool-call-parser hermes` to
  vLLM (needed for Qwen tool calls; pass `llama3_json` as the 2nd sbatch arg for
  Llama models) and fixes the scorer invocation to use the underscore model name.
- Generation resumes from partial results — if the job hits the time limit, just
  resubmit it.
- Ablations: `WTB_SC_N=1 sbatch run_wtb_vllm.sh ...` isolates the triage-prompt
  effect; comparing against `score/` isolates the total effect.

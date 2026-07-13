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

### Robustness to server-side parse failures

At `WTB_SC_TEMP=0.8` a small model sometimes emits a truncated/malformed tool
call that vLLM's `hermes` parser cannot parse; vLLM then returns the raw text as
`content` with no `tool_calls` (this is what the `[vLLM Patch Warning] Failed to
parse tool call...` lines report — harmless, non-fatal). If left as-is that text
would count as an "answer" vote and could outvote a correct tool-call candidate.
So before voting, each candidate is normalized (`_normalize_response`):

- If the content is a recoverable tool-call attempt (e.g. a `<tool_call>` block or
  a bare `{"name": ...}` JSON, possibly truncated), it is repaired via brace
  balancing and converted back into a real tool call so it joins the tool cluster.
- If it is an unrecoverable attempt, its content is blanked so the candidate
  **abstains** from the vote instead of masquerading as a valid text answer.

This is done in portable handler code (not by patching vLLM), so it also salvages
calls the server-side patch drops. The single-sample path (`WTB_SC_N=1`) returns
the model's response unchanged.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `WTB_SC_N` | 5 | candidates per step; `1` disables voting entirely |
| `WTB_SC_TEMP` | 0.8 | temperature of the diversity samples |

## Running on SOL

See `SOL_VLLM_GUIDE.md` for the full interactive-session walkthrough. In short,
after starting the vLLM server (with `--enable-auto-tool-choice --tool-call-parser
hermes`, and `python3 patch_vllm.py` applied once):

```bash
cd WildToolBench/wild-tool-bench
cp .env.example .env    # one-time; the gitignored .env is not carried by git pull
python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct --num-threads 4 --result-dir result_consensus
python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct --result-dir result_consensus --score-dir score_consensus
```

Notes:
- Results go to **`result_consensus/` + `score_consensus/`** so the committed
  baseline in `result/` + `score/` stays intact for comparison. (The generator
  skips ids that already have results, so re-using `result/` would silently skip
  the whole run.)
- The scorer takes the **underscore** form of the model name (`Qwen_Qwen2.5-7B-Instruct`).
- Generation resumes from partial results — if the job hits the time limit, just rerun it.
- Ablation: prefix inference with `WTB_SC_N=1` and use fresh result/score dirs to
  isolate the triage-prompt effect; comparing against `score/` isolates the total effect.

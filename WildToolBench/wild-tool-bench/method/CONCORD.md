# CONCORD: conservative contract-dominance

CONCORD is the current `demo2` method. It keeps the useful mechanical parts
of IGAR-v24 and the anchor protection of GAVEL-v2, but moves all state and
decision logic outside the model prompt.

The name stands for **CONservative CONtract-based Runtime Dominance**.

## What the result evidence actually says

The strongest official Qwen2.5-7B result committed so far is IGAR-v24:

| Method | Correct tasks | Complete sessions |
|---|---:|---:|
| Clean `demo2` baseline | 354/1024 | 4/256 |
| GAVEL-v1 | 362/1024 | 5/256 |
| GAVEL-v2 | 385/1024 | 8/256 |
| IGAR-v24 | **404/1024** | **12/256** |

The previously documented 17-session number was a saved-trace
counterfactual, not a fresh inference result. It is not an official benchmark
claim. CONCORD also has no fresh score yet; run it into a new result directory.

The IGAR-v24 layer-3 count of 75 means 75 fourth turns were correct. It does
not mean 75 sessions failed only at turn 3. Its exact survival chain is:

```text
turn 0 correct:       141
turns 0-1 correct:     64
turns 0-2 correct:     24
all four correct:      12
```

Only 12 sessions have pattern `1110`. Across all positions, 34 sessions have
exactly one failed turn. Repairing all 34 with no regression would yield 46
sessions, but that is an upper-bound diagnostic rather than a prediction.

## Why the earlier mechanisms plateaued

IGAR-v24 injected a verified ledger, surfaced facts, and dialogue-state notes
as additional system messages. That helped long-range binding, but conflicts
with the no-added-system-instructions constraint. Its state was also mostly a
bag of slots: it could not reliably represent ordered references, `last two`
plus `respectively`, unfinished conditional clauses, or the difference
between a necessary and merely possible optional argument.

GAVEL-v2 removed the prompt additions and protected text anchors. That fixed
many Chat and Clarify regressions, but its receipt gate made a logical error:
it treated failure to prove the source of a present required argument as
proof that the argument was absent. In the committed run, 142 schema-authored
questions were emitted; 128 landed on turns whose next action was a tool call.

CONCORD uses a different asymmetry:

> Trust complete required structure; minimize unsupported optional
> commitments; intervene only on a positive defect.

## Runtime boundary

The handler reads only:

```text
messages
tools
```

It never reads `answer_list`, task type, task id for decision-making, labels,
scores, candidate gold paths, or evaluator output. The task id and turn are
used only as labels in the audit log.

Every model request receives the same native messages and tool list. CONCORD
does not append a system, user, assistant, ledger, state, critique, or repair
message. It never uses `tool_choice="required"`.

## The mechanism

1. **Endow the anchor.** Qwen's greedy response is the default decision.
2. **Trust complete required values.** A known tool with valid JSON, valid
   declared types, all required keys, explicit authorization for side
   effects, and no unresolved guard is returned. A weak provenance heuristic
   cannot veto it.
3. **Build an external contract.** From visible messages and schemas,
   CONCORD records completed calls, outstanding request clauses, ordered
   entities, same/previous/initial references, relative years, and stable
   slots. This structure is never shown to the model.
4. **Resolve before asking.** Deterministic repairs cover cases such as
   `the word at the beginning`, `last three years`, and record arrays bound by
   `last two ... 10 and 50 respectively`. If a required field is still truly
   absent, clarification remains available.
5. **Minimize Boolean commitments.** An optional Boolean is removed unless
   its meaning is supported by the active or same-tool goal. Explicit
   exclusions and broad requests for complete/other information are
   preserved.
6. **Collapse exact duplicates.** Correlated generations cannot gain weight
   by repeating an identical call.
7. **Repair text only by agreement.** A text anchor is reconsidered only when
   the contract proves an executable clause remains. Two unchanged-input
   native generations must agree on the same tool names and required
   arguments, and their bundle must cover every provable residual clause.
   Otherwise the text anchor survives.
8. **Select by dominance, not vote count.** Among agreeing complete bundles,
   prefer fewer optional commitments, lower irreversible risk, fewer calls,
   and a deterministic canonical tie-break.

This differs from IGAR's whole-output plurality and GAVEL's call-by-call
receipt market. Agreement is component-level over required commitments, while
the final decision remains an actual native model generation.

## Pull and verify

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
git switch demo2
git pull --ff-only origin demo2

python3 -m unittest method.test_concord method.test_gavel -v

WTB_METHOD=concord python3 -c "
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"
```

The last command must print `ConcordHandler`.

## Run the full benchmark

Leave the configured OpenAI-compatible model endpoint running. Use a new
directory; the WTB runner resumes existing result directories and can
otherwise silently mix methods.

```bash
WTB_METHOD=concord python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 8 \
  --result-dir result_concord_v1

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_concord_v1 \
  --score-dir score_concord_v1
```

For a matched clean-baseline and method run:

```bash
bash method/run_full.sh --tag concord_v1
```

The script creates separate baseline and CONCORD result/score directories,
checks routing before generation, and reports fixed/broken turns afterward.
GAVEL-v2 remains available with `bash method/run_full.sh --arm gavel`.

## Runtime controls

| Variable | Default | Meaning |
|---|---:|---|
| `WTB_CONCORD_CANDIDATES` | `3` | Total native generations when repair opens |
| `WTB_CONCORD_TEMPERATURE` | `0.2` | Temperature for the two shadow generations |
| `WTB_CONCORD_REQUEST_TIMEOUT` | `600` | Per-request timeout in seconds |

Chat, final-answer, and complete-tool anchors normally use one request. A text
anchor with a provable residual tool obligation or a structurally defective
tool anchor can use up to three. All calls use the same model; eight runner
threads do not create eight copies of the weights.

Every decision emits one `[CONCORD]` JSON record. Preserve the generation log
with the result because it is needed to distinguish anchor preservation,
reference repair, agreement-based recovery, genuine clarification, and
fallback.

## Validation before the first run

- Model-free tests cover the evaluator firewall, unchanged request boundary,
  required-value trust, optional Boolean removal and preservation, ordinal
  reference repair, relative years, `last two ... respectively`, exact
  duplicate collapse, residual-clause closure, agreeing recovery, and
  disagreement fallback.
- Saved-trace replay inspected every emitted tool step in the clean baseline,
  GAVEL-v2, and IGAR-v24 traces. The current normalizer changed 105, 29, and
  56 already-failing steps respectively, and changed **zero** steps whose
  saved argument check or whole task was correct.
- These replays validate conservative editing only. They do not predict a
  fresh score because changing an action also changes later observations and
  model inputs.

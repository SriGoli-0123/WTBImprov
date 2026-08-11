# GAVEL-v2: anchor sovereignty

> **Superseded experiment:** GAVEL-v2 is retained as an ablation. CONCORD was
> evaluated after it, and the current experiment is [PRISM](PRISM.md), which
> targets the paper's self-conditioning diagnosis using exact evidence views.

GAVEL stands for **Guarded, Axiomatic Verification and Execution
Lexicography**.  V2 is the evidence-driven correction to the first GAVEL run.
It keeps the useful structure of IGAR-v24—argument receipts, conservative
selection, and dependency awareness—without adding a ledger, facts, dialogue
state, or repair instructions to the model prompt.

The runtime boundary is strict:

- the model receives the ordinary WTB messages and complete tool list;
- GAVEL reads only `messages` and `tools`;
- `answer_list`, task types, labels, scores, and evaluator output are never
  read by the handler;
- a meaningful text anchor is returned immediately and unchanged;
- a mechanically valid tool anchor is returned as one indivisible bundle;
- alternative generations reuse the exact messages and tools, but are opened
  only to rescue a provably defective tool anchor;
- proposal counts do not vote;
- all state is local to a request, so eight concurrent threads cannot overwrite
  each other's deadlines or decisions.

## What failed in v1

The committed v1 run scored 362/1024 tasks and 5/256 sessions.  Its forced
tool branch produced 274 outputs containing both the retained text anchor and
an inserted tool call; none scored correct.  All 26 turns that changed from
correct in the clean baseline to wrong had this signature.

An offline diagnostic suppressed every inserted call, including cases where a
tool really was needed, and reran the official scorer over the saved trace.  It
produced 483/1024 tasks and 13/256 sessions.  Replaying the complete v2 policy
on the same trace—text sovereignty plus selective optional normalization—gave
500/1024 tasks and 17/256 sessions.  These are counterfactual regression tests,
not fresh benchmark claims.  The evaluator was used only after generation for
scoring and is not available to the runtime method or model.

The vLLM `tool_choice="required"` 400 response was a symptom of the discarded
branch, not the cause of the low score.  V2 never makes that request.

## The v2 mechanism

The stock response is the **anchor** and owns the decision by default:

1. **Text sovereignty.** Nonempty text is returned after one request.  A
   sampled call can never replace an answer or clarification.
2. **Tool-anchor ownership.** A mechanically viable original tool bundle is
   returned whole.  It is not sampled against, re-ranked, split, or extended.
3. **Certified rescue only.** Extra exact-input proposals are generated only
   when the tool anchor has a hard defect such as an unknown tool, invalid
   shape, missing required value, unresolved guard, completed duplicate, or
   missing authorization.
4. **Minimum optional commitment.** An optional documented default is omitted
   unless the current user explicitly selected it.  Optional decoration is
   also omitted for an underspecified “another/new” object.
5. **Ambiguous selector gate.** A state-changing call cannot use “one of them”
   as authorization when the current user did not identify which object.

A rescued call is admitted only when its tool and argument shape are valid,
its required values have visible provenance, any conditional guard is ready,
the call is authorized, and an identical completed result is not already
available.  A fresh/latest/real-time read is allowed to repeat.

For rescue proposals only, GAVEL applies this order exactly:

1. cover the most current user obligations;
2. make the least irreversible commitment;
3. make the fewest calls;
4. use a deterministic canonical tie-break.

If no call is executable but a relevant proposal lacks required information,
GAVEL asks for the field set that unlocks the most obligations at the lowest
information/sensitivity cost.  The question is rendered from schema field
descriptions in code; no clarification prompt is sent to the model.

If an anchor call is a completed duplicate, an unresolved conditional side
effect, an ambiguous state change, or a constructive action triggered only by
topic overlap, GAVEL uses the model's ordinary text branch through
`tool_choice="none"`.  It does not append an instruction.

## Pull and verify

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
git switch demo2
git pull --ff-only origin demo2

python3 -m unittest method.test_gavel -v

python3 -c "
import os
os.environ['WTB_METHOD'] = 'gavel'
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"
```

The last command must print `GavelHandler`.

## Run GAVEL directly

Leave the configured OpenAI-compatible model endpoint running, then run:

```bash
WTB_METHOD=gavel python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 8 \
  --result-dir result_gavel_v2

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_gavel_v2 \
  --score-dir score_gavel_v2
```

Use a new result directory.  The runner resumes existing directories, so
reusing a baseline or Grounded directory would silently mix methods.

## Run matched baseline and GAVEL arms

```bash
bash method/run_full.sh --tag gavel_v2
```

This creates:

```text
result_gavel_v2_baseline/
result_gavel_v2_gavel/
score_gavel_v2_baseline/
score_gavel_v2_gavel/
logs_gavel_v2/
```

It also performs the fixed/broken comparison.  The legacy Grounded wrapper is
still available with `bash method/run_full.sh --arm grounded`.

## Runtime controls

| Variable | Default | Meaning |
|---|---:|---|
| `WTB_GAVEL_CANDIDATES` | `2` | Total candidates when rescuing an invalid tool anchor |
| `WTB_GAVEL_TEMPERATURE` | `0.2` | Temperature for rescue proposals |
| `WTB_GAVEL_REQUEST_TIMEOUT` | `600` | Per API request timeout in seconds |

The anchor always uses the temperature supplied to the WTB runner, normally
zero.  Text and valid-tool turns make exactly one request.  More candidates
increase cost only on invalid tool anchors, and repeated identical proposals
never gain decision weight.

Every decision is printed as a single `[GAVEL]` JSON line.  It records the
anchor mode, proposals, receipts or rejection reasons, selected calls, and
decision without exposing evaluator information.  Keep these logs: they make
the ablation and failure attribution reproducible.

## Validation

- Twenty-one model-free unit tests cover text sovereignty, one-request behavior,
  tool-anchor ownership, authorization, missing fields, novelty, documented
  defaults, ambiguous selectors, exact result matching, conditional side
  effects, rescue selection, parallel-bundle preservation, minimum-information
  clarification, and the evaluator firewall.
- Suppressing every v1 forced-tool override scored 483/1024 tasks and 13/256
  sessions; replaying the complete v2 policy scored 500/1024 and 17/256.
- The v2 argument normalizer was replayed over 1,218 clean-baseline tool calls.
  It changed 149 calls from already-failing or unscored steps and **zero calls
  whose saved argument check was correct**.
- This branch still requires a fresh v2 inference run.  No counterfactual is
  reported as an official model result.

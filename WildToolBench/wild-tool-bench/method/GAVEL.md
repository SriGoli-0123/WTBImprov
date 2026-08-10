# GAVEL: a clean successor to IGAR-v24

GAVEL stands for **Guarded, Axiomatic Verification and Execution
Lexicography**.  It keeps the useful structure of IGAR-v24—state, argument
receipts, conservative selection, and dependency awareness—without adding a
ledger, facts, dialogue state, or repair instructions to the model prompt.

The runtime boundary is strict:

- the model receives the ordinary WTB messages and complete tool list;
- GAVEL reads only `messages` and `tools`;
- `answer_list`, task types, labels, scores, and evaluator output are never
  read by the handler;
- alternative generations reuse the exact messages and tools and differ only
  through the API's decoding/tool-choice controls;
- proposal counts do not vote;
- all state is local to a request, so eight concurrent threads cannot overwrite
  each other's deadlines or decisions.

## The mechanism

The stock response is the **anchor**.  A candidate call is admitted only when
its tool and argument shape are valid, its required values have visible
provenance, any conditional guard is ready, the call is authorized, and an
identical completed result is not already available.  A fresh/latest/real-time
read is allowed to repeat.

For a compatible call bundle `S`, GAVEL applies this order exactly:

1. cover the most current user obligations;
2. make the least irreversible commitment;
3. retain the most anchor calls;
4. make the fewest calls;
5. use a deterministic canonical tie-break.

A mechanically viable anchor bundle is indivisible.  This is important:
splitting a correct parallel answer to re-rank its individual calls creates
regressions.  Extra sampled calls can extend an anchor only when they carry
strict receipts and the current request explicitly has capacity for another
action.

If no call is executable but a relevant proposal lacks required information,
GAVEL asks for the field set that unlocks the most obligations at the lowest
information/sensitivity cost.  The question is rendered from schema field
descriptions in code; no clarification prompt is sent to the model.

If a proposed call is a completed duplicate, an unresolved conditional side
effect, or a constructive action triggered only by topic overlap, GAVEL uses
the model's ordinary text branch through `tool_choice="none"`.  It does not
append an instruction.

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
  --result-dir result_gavel

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_gavel \
  --score-dir score_gavel
```

Use a new result directory.  The runner resumes existing directories, so
reusing a baseline or Grounded directory would silently mix methods.

## Run matched baseline and GAVEL arms

```bash
bash method/run_full.sh --tag gavel_v1
```

This creates:

```text
result_gavel_v1_baseline/
result_gavel_v1_gavel/
score_gavel_v1_baseline/
score_gavel_v1_gavel/
logs_gavel_v1/
```

It also performs the fixed/broken comparison.  The legacy Grounded wrapper is
still available with `bash method/run_full.sh --arm grounded`.

## Runtime controls

| Variable | Default | Meaning |
|---|---:|---|
| `WTB_GAVEL_CANDIDATES` | `2` | Anchor plus exact-input proposal count |
| `WTB_GAVEL_TEMPERATURE` | `0.2` | Temperature for non-anchor proposals |
| `WTB_GAVEL_EXPLORE_TEXT` | `1` | Explore a forced-tool proposal when the anchor is text |
| `WTB_GAVEL_REQUEST_TIMEOUT` | `600` | Per API request timeout in seconds |

The anchor always uses the temperature supplied to the WTB runner, normally
zero.  More candidates increase cost, but repeated identical proposals never
gain decision weight.

Every decision is printed as a single `[GAVEL]` JSON line.  It records the
anchor mode, proposals, receipts or rejection reasons, selected calls, and
decision without exposing evaluator information.  Keep these logs: they make
the ablation and failure attribution reproducible.

## Validation completed before the first model run

- Fourteen model-free unit tests cover authorization, missing fields, novelty,
  exact result matching, conditional side effects, risk ordering, coverage,
  minimum-information clarification, and the no-added-instructions boundary.
- The deterministic verifier was replayed over all 857 tool-producing steps in
  the committed full baseline.  It changed 66 errored steps and **zero steps
  that the committed scorer marked correct**.  This is a regression audit, not
  a new benchmark result; GAVEL still needs a fresh full inference run.

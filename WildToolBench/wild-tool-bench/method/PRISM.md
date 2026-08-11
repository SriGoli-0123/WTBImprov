# PRISM: reset the policy without adding a prompt

PRISM is the current `demo2` experiment. It replaces CONCORD's unsafe rule
that every complete-looking tool call should be preserved.

PRISM stands for **Policy Reset through Invariant Sliced Memory**. The name is
less important than the idea: let the same model look at the same available
evidence from a few different distances, and trust decisions that remain
stable.

## Honest result status

The best official Qwen2.5-7B result anywhere in this repository is still
IGAR-v24:

| Method | Correct tasks | Complete sessions |
|---|---:|---:|
| Clean `demo2` baseline | 354/1024 | 4/256 |
| GAVEL-v2 | 385/1024 | 8/256 |
| CONCORD-v1 | 375/1024 | 6/256 |
| IGAR-v24 | **404/1024** | **12/256** |

PRISM has not been run on the full benchmark yet. It is therefore not claimed
as an improvement until a fresh result exceeds 404 tasks and 12 sessions.

CONCORD did improve some isolated numbers over IGAR-v24: Single-Tool rose from
97 to 106 correct, optimal parallel/mixed execution rose from 32 to 39, and
progress rose from 179 to 187 completed steps. But it lost 22 Chat tasks and 16
Clarify tasks. Those losses reduced complete sessions from 12 to 6.

## What went wrong before

The WildToolBench paper says that long conversations create
**self-conditioning**. After using a tool, a model tends to continue using
tools even after the user has switched to an explanation, ordinary chat, or a
request with missing information. Historical messages also dilute attention to
the current task.

CONCORD made this worse by protecting any tool call that had all required JSON
fields. A guessed field can still be perfectly valid JSON. For example, the
model guessed `Credit Card` when the user had not supplied a payment method.

IGAR-v24 handled history better, but it added ledger and dialogue-state system
messages. It also built those messages from benchmark answer history. That
violates the constraints for the current project and is not a general runtime
solution.

## The mechanism in simple terms

PRISM makes three native model requests for each decision:

1. **Full view:** the ordinary complete conversation.
2. **Reset view:** the system date and the current task, including any tool
   results or clarification replies already produced inside that task.
3. **Focused view:** PRISM retrieves the most relevant exact earlier exchange
   and places it beside the current task. It does this even when a terse human
   follow-up such as “China” or “Convert to EUR” contains no obvious reference
   word.

Every message in these views is copied from the model's existing input. PRISM
does not write or append an instruction, summary, ledger, critique, example, or
hidden hint. The tool list is unchanged.

The decision rule is deliberately small:

- Two views must agree before replacing the full-history tool/text choice.
- The focused view is the tie-breaker: for a genuine continuation it usually
  agrees with the full view; after a topic switch it can agree with the clean
  reset view because irrelevant older exchanges have been removed.
- After a tool result or clarification reply arrives inside the current task,
  that task slice is self-contained. PRISM then gives it two votes so an older
  session cannot reopen a finished action.
- On a first turn, where there is no earlier exchange to focus, two unchanged
  current-task samples provide the same majority check.
- Tool calls count as agreement only when the tool set and every required
  value match. PRISM then prefers fewer unsupported optional values and lower
  irreversible risk.
- An exact completed call is not repeated unless the user explicitly asks to
  repeat it.
- A required identity or user choice that is genuinely absent—payment method,
  location, account, file, recipient, and similar fields—causes a question
  instead of an invented value.
- Once the user supplies the missing value, the current-task view retains the
  complete clarification chain and allows the tool call to continue.

This directly addresses WildToolBench's three concerns: policy switching,
hidden intent across turns, and multi-step continuation.

## Constraint boundary

PRISM is:

- benchmark-agnostic;
- model-agnostic;
- training-free and RL-free;
- evaluator-free at runtime;
- free from added system or user instructions;
- based only on `messages` and `tools` for decisions.

It never reads `answer_list`, task type, turn subtype, labels, scores, gold tool
paths, or evaluator output. Benchmark IDs and turn indices appear only in the
human-readable `[PRISM]` diagnostic log.

## Pull and verify

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
git switch demo2
git pull --ff-only origin demo2

python3 -B -m unittest method.test_prism method.test_concord method.test_gavel -v

WTB_METHOD=prism python3 -c "
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"
```

The last command must print `PrismHandler`.

## Full run

Use a brand-new directory. Reusing a result directory can silently mix old and
new outputs.

```bash
WTB_METHOD=prism python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 8 \
  --result-dir result_prism_v1

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_prism_v1 \
  --score-dir score_prism_v1
```

Or run a matched clean baseline and PRISM arm:

```bash
bash method/run_full.sh --tag prism_v1
```

PRISM normally makes three model requests per decision, so it will take longer
than the clean baseline. The runner's eight threads still share one loaded
model; they do not load eight copies.

## Controls

| Variable | Default | Meaning |
|---|---:|---|
| `WTB_PRISM_CANDIDATES` | `3` | Evidence-view decisions per step |
| `WTB_PRISM_TEMPERATURES` | `0.15,0.45,0.7,0.9` | Temperatures for non-anchor decisions |
| `WTB_PRISM_REQUEST_TIMEOUT` | `600` | Timeout for each request |

Keep the defaults for the first official run. Changing them before obtaining a
matched result would make diagnosis harder.

## Validation performed before release

- The PRISM, CONCORD, and GAVEL model-free suites pass together.
- Tests cover policy reset, reference retention, multi-step clarification,
  missing-choice questions, explicit choices, repeat suppression, unstable
  disagreement, evaluator isolation, and the no-added-message boundary.
- The missing-information gate was replayed over every saved tool step from
  IGAR-v24, GAVEL-v2, and CONCORD-v1. It flagged only previously failing steps
  and changed zero saved correct steps.

The saved replay is a safety check, not a predicted score. The three evidence
views create new generations, so only a fresh 256-session run can establish the
actual result.

# IGAR-v26: documented continuity without changing the prompt

IGAR-v26 is a conservative upgrade of v25. Its main rule is simple:

> Keep the model's exam paper unchanged; keep the bookkeeping outside it.

## What the model receives

Exactly the normal WildToolBench input:

1. one system message containing only `Current Date: ...`;
2. the native user, assistant, and tool-result history;
3. the current user message; and
4. the supplied tool documents (name, description, and JSON parameter schema).

V26 does not insert a ledger, dialogue-state summary, audit request, worked
example, evaluation label, or reference answer. It does not change temperature,
train the model, or use RL.

## What stays outside the model

V26 compiles the supplied tool schemas into a small controller-side contract.
It uses that contract in two narrow cases.

### 1. An answered clarification must make progress

When the model asks for a documented field, V26 remembers the tool and field.
If the user gives useful information but the model merely repeats the same
question or stops, V26 retries the *same untouched conversation* with that one
documented tool selected. The retry is accepted only if the call exists in the
supplied tools, passes its JSON schema, introduces no unsupported required
string, and is not a duplicate call.

Deferrals such as “wait a minute,” uncertainty, cancellations, and genuinely
new missing fields are left alone. This is a progress rule, not a guesser.

### 2. A reference should not silently change an established value

For phrases such as “that one,” “same,” or “again,” V26 compares a proposed
call with the latest already executed historical call to the same tool. It
repairs only a changed scalar that the user did not mention. If the user gives
a new value or says “another,” “different,” or “instead,” the proposal is kept.

This turns semantic checking into difference detection: the controller does
not solve the task from scratch; it notices one unsupported change and restores
the observed value.

## Safety boundary

- Current-task gold answers and scorer outputs are never read by V26.
- Assistant prose is not accepted as evidence for argument values (the v25
  provenance rule).
- Historical calls count only when a matching tool-result message proves they
  were executed.
- The ordinary model result remains sovereign unless a narrow mechanical check
  proves the continuity violation.
- If a forced-tool retry is unsupported by a server, V26 keeps the ordinary
  result rather than failing the session.

These rules are benchmark-agnostic: they need only a conversation, ordinary
tool documentation, and tool results. They contain no WTB IDs, task labels, or
domain-specific tool names.

## Run

Start the same OpenAI-compatible model server you normally use. In the runner
terminal:

```bash
cd ~/wildtoolbench_workspace_vllm/WildToolBench/wild-tool-bench
conda activate phase1_env

git switch demo2
git pull origin demo2
git branch --show-current

unset WTB_SC_N WTB_SC_TEMP WTB_ENTITY_LABELS WTB_ASK_GATE
export WTB_METHOD=igar_v26

python3 -B -c "
from wtb.model_handler.handler_map import HANDLER_MAP
print(HANDLER_MAP['Qwen/Qwen2.5-7B-Instruct'].__name__)
"

python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-threads 8 \
  --result-dir result_igar_v26

python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_igar_v26 \
  --score-dir score_igar_v26
```

The verification line must print `IGARV26Handler`. Use a fresh result directory;
the runner resumes and may otherwise reuse outputs produced by another method.

No temperature override is part of these commands.

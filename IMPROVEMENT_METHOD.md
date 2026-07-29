# Contrastive Action Verification (CAV)

Status: implemented on `demo`, not yet scored.

## Goal

Increase WildToolBench session accuracy by improving the model's general
multi-turn tool-use decisions without:

- changing the system prompt;
- training or updating model weights;
- using RL;
- using WTB task labels, expected actions, scores, or evaluator feedback;
- adding benchmark-specific keywords or rules.

The model still receives exactly:

1. `system`: `Current Date: <english_env_info>`;
2. the native visible conversation;
3. the current user message;
4. the supplied detailed tool schemas.

## Simple idea

Let the model answer normally, then verify that a proposed action:

1. was caused by the latest user/tool information;
2. is not merely continuing prior tool-call syntax;
3. uses values traceable to visible information;
4. is schema-valid and executable now;
5. has not already completed, unless the latest event causally re-authorizes it.

The normal response is preserved unless a contradiction is mechanically
demonstrated.

## Algorithm

For each step:

1. **Anchor generation** — generate normally on the untouched messages and tools.
2. If the anchor is text, preserve it.
3. If the anchor contains tool calls, run two diagnostic model views concurrently:
   - **without latest event**: blank the latest user content, or remove the
     latest completed tool execution;
   - **without prior action syntax**: remove earlier assistant tool-call syntax
     while retaining the raw visible results.
4. Construct **per-argument causal receipts**. A derived value is causally
   supported only when it:
   - appears in the anchor and syntax-neutral response; and
   - disappears when the latest external event is removed.
5. Build the **executable frontier**:
   - retain known tools with valid JSON and schema types;
   - require every required argument to have direct or causal provenance;
   - remove unsupported optional arguments and unknown keys;
   - suppress exact calls that already produced a visible tool result, unless
     the latest event causally re-authorizes the repetition;
   - emit all remaining independent executable calls together.
6. If no call remains, or both counterfactuals prove action momentum, request
   model-authored text on the original conversation with `tool_choice="none"`.
   No corrective prompt is added.
7. If a diagnostic or text fallback fails, preserve the usable anchor response.

## Evaluator firewall

`CAVController` accepts only:

```python
{"messages": [...], "tools": [...]}
```

It rejects evaluation-only or unknown keys, including:

- `answer_list` / `english_answer_list`;
- task and test IDs;
- gold/expected values;
- score fields;
- tool-call graphs;
- prior answer-list objects.

The WTB evaluator still uses the current task's answer list outside CAV to score
the emitted action and provide simulated tool observations. This data is never
passed to the controller or model.

WTB's official teacher-forced prior-turn history remains ordinary visible
conversation history. CAV does not convert it into ledger, state, policy, or
extra system messages.

## Why this is benchmark-agnostic

CAV does not classify a turn as WTB Chat, Clarify, Single, or Multi. It asks
four universal deployment questions:

- What changed?
- What evidence supports each value?
- Which actions are executable now?
- Which actions already completed?

These questions also apply to real conversations containing corrections,
references, interruptions, changing goals, missing details, and dependent tools.

## Runtime and configuration

`WTB_CAV_WORKERS` defaults to `8`. The present implementation uses at most two
concurrent diagnostic requests per tool-emitting step; the value is a ceiling,
not eight model copies. A single shared vLLM model server can batch the requests.

CAV costs:

- one normal generation for every step;
- up to two concurrent diagnostic generations for a tool anchor;
- one text-only generation only when intervention is proven.

## Files

- `wtb/model_handler/cav.py` — standalone controller and firewall;
- `wtb/model_handler/base_handler.py` — native WTB history plus runtime-only CAV integration;
- `wtb/model_handler/api_inference/oai.py` — model-authored text fallback;
- `tests/test_cav.py` — benchmark-independent unit tests.

## Validation and scoring discipline

Implemented tests verify:

- evaluation-only data is rejected;
- the original date-only system prompt is preserved;
- supported anchors remain unchanged;
- derived representations require counterfactual support;
- action momentum falls back to model-authored text;
- stale exact repeats are blocked while causally requested repeats remain allowed;
- unsupported dependent calls are removed from the executable frontier;
- unknown and unsupported optional arguments are removed.

Do not tune CAV on WTB results if WTB is intended to remain an untouched test.
The existing repository has already been repeatedly developed against all 1,024
WTB tasks, so the next WTB score must be reported as exploratory/development-set
performance. Confirm generality on a frozen, independent messy-intent test set.

Primary target: session accuracy. Also report task accuracy, accomplishment
progress, optimal-path accuracy, task-type breakdown, baseline fixes, and
baseline regressions; a session gain alone does not prove every other metric
improved.

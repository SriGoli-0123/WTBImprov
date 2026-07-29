# Counterfactual Obligation Graph with Frontier Synthesis (COG-FS)

Status: implemented on `demo`; local unit-tested; not yet scored on
WildToolBench.

## Objective

The primary objective is to exceed the current best result of **12/256 correct
sessions** while improving, rather than trading away, the underlying task
metrics. The current best recorded task score is **404/1024**.

COG-FS is designed for the problem WildToolBench exposes: real requests are
messy, multi-turn, partially specified, referential, and often contain several
independent or dependent obligations. A model can understand most of a request
yet omit one action, continue an old action, invent a parameter, or choose text
when execution was required. Since a session is correct only when all four
turns are correct, one such local mistake destroys the session.

The method does not:

- change or improve the system prompt;
- train or update the model;
- use RL, fine-tuning, demonstrations, or a second learned model;
- use WTB labels, task types, expected calls, answer lists, graphs, or scores;
- encode benchmark-specific domains, tool names, or keywords.

The model still receives only:

1. `system`: `Current Date: <english_env_info>`;
2. the native visible conversation, including ordinary prior tool results;
3. the current user message;
4. the detailed tool schemas supplied for that session.

## Simple idea

Ask the model once on the untouched request. Then remove small pieces of the
request and ask again.

If removing one clause makes a particular tool call disappear, that is causal
evidence that the clause created an obligation for that call. A call discovered
only in a reduced view must also survive a confirmation on the untouched
conversation with only its own tool schema visible. COG-FS combines all
confirmed, currently executable obligations into one frontier.

This is not majority voting. The outputs are not interchangeable samples: each
shadow has had a specific source of evidence removed. The method learns from
the direction of the change, not from which answer is most popular.

## Algorithm

For each model step:

1. **Untouched anchor** — generate normally from the original messages and all
   supplied tool schemas.
2. **General clause discovery** — split the latest user text at punctuation,
   newlines, and ordinary coordination markers such as “also” and “then.” At
   most three spans are used. There is no task classifier or domain vocabulary.
3. **Subtractive shadows** — in parallel, generate on:
   - the conversation without the latest external event;
   - the conversation with prior assistant tool-call syntax removed but its
     visible results retained;
   - one view for each latest-user clause, with only that clause blanked.
4. **Obligation graph** — collect calls from the anchor and shadows. A call has
   causal support when it is present with a clause/event and absent without it.
   The no-latest-event view is a control and can never introduce a new action.
5. **Confirmation of recovered calls** — a call found only in a shadow is
   admitted only if the exact same call is regenerated on the untouched
   conversation when its own schema is isolated.
6. **Conservative text-anchor projection** — if the anchor is text, COG-FS may
   test one lexically relevant, required-argument tool in isolation. It admits
   the call only if:
   - it appears on the full conversation;
   - it disappears without the latest event; and
   - at least one required value is directly supported by the latest user
     message.
   This recovery path is deliberately narrow so an explanatory Chat request
   is not converted into an invented action.
7. **Executable-frontier synthesis** — emit every unique confirmed call whose:
   - tool exists in the supplied registry;
   - arguments parse and satisfy the supplied schema;
   - required values have direct conversational or counterfactual support;
   - optional string values are explicitly supported;
   - exact action has not already completed, unless the new event
     re-authorizes it.
   Unknown keys and unsupported optional values are removed.
8. **Safe fallback** — if no tool call remains, preserve the model's original
   text. If a tool anchor becomes non-executable, request model-authored text on
   the untouched conversation with tool use disabled. No corrective prompt is
   added. If diagnostics fail, preserve the usable anchor.

## Why this targets session accuracy

The earlier methods mostly edited one proposed call. That cannot recover an
independent call the model never proposed, and improvements in one category can
break another. COG-FS instead tries to reconstruct the complete set of
obligations caused by the current request:

- clause masks expose omitted parallel actions;
- the latest-event control distinguishes a new request from action momentum;
- syntax neutralization tests whether prior call formatting is driving the
  answer;
- per-tool confirmation protects against calls manufactured by an ablation;
- frontier synthesis emits independent executable calls together;
- provenance and schema checks block unsupported or malformed calls.

These are general interaction invariants, not WTB answer patterns. They also
apply to assistants facing corrections, topic switches, pronouns, partial
information, dependent actions, and several requests in one message.

## Evaluator firewall

`COGFSController` accepts exactly:

```python
{"messages": [...], "tools": [...]}
```

Unknown keys are rejected. Therefore answer lists, expected actions, task/test
IDs, score fields, tool-call graphs, and evaluator state cannot reach COG-FS or
the model. WTB keeps those objects in its outer evaluation loop, where they
belong.

The official teacher-forced prior-turn history remains visible only as ordinary
conversation history. COG-FS does not turn it into a state ledger, hidden
policy, or extra system message.

## Runtime

`WTB_COGFS_WORKERS` defaults to `8`. The value is both the concurrency ceiling
and the hard maximum number of model generations per decision:

- one untouched anchor;
- up to seven diagnostic, projection, confirmation, or text-fallback calls.

All requests use one shared vLLM server and one loaded model. Eight workers do
not load eight model copies. Identical shadow views are deduplicated before
generation.

## Files

- `wtb/model_handler/cogfs.py` — controller, discovery, confirmation, frontier,
  and firewall;
- `wtb/model_handler/base_handler.py` — strict runtime integration;
- `wtb/model_handler/api_inference/oai.py` — unchanged-input text fallback;
- `tests/test_cogfs.py` — benchmark-independent behavior and safety tests;
- `tests/test_cav.py` and `wtb/model_handler/cav.py` — retained historical CAV
  tests and implementation, no longer active.

## Validation and scoring discipline

Local tests cover:

- the date-only system prompt and strict messages/tools firewall;
- preservation of supported text and tool anchors;
- recovery and union of an omitted independent call;
- rejection of an unconfirmed shadow call;
- conservative recovery from a text anchor without converting unsupported Chat;
- support for derived required values and non-string intent switches;
- removal of unsupported optional strings;
- completed-action blocking and explicit re-authorization;
- grounding in prior visible tool results;
- the hard eight-generation ceiling.

COG-FS is **not yet WTB-scored**, so no improvement is claimed. A successful
promotion requires more than 12 correct sessions and more than 404 correct
tasks, with task-type, accomplishment-progress, and optimal-path metrics checked
for regressions. Use a fresh result directory because WTB resumes existing
outputs.

Because this repository has already been repeatedly developed against all WTB
tasks, the next WTB run is exploratory/development-set evidence. Benchmark-
agnostic generality should also be checked on a frozen independent set of messy
multi-turn requests whose evaluator is never exposed to the model.

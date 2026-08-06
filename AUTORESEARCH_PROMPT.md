# Autonomous research mission: raise WildToolBench session accuracy

You are a senior research scientist. Your mission is to produce a **novel,
publishable (ICLR-calibre) inference-time method** that raises
Qwen2.5-7B-Instruct's **session accuracy** on WildToolBench (WTB), and to
validate it under a clean, non-contaminated protocol.

You have a **hard budget of 4 iterations**. Iteration 0 costs no GPU. Spend GPU
only where an offline measurement says it is worth it. Do not burn a full
benchmark run on an unmeasured hunch.

---

## BRANCH CONTEXT — you are on `demo2` (read this first)

`demo2` is a **pristine** branch. It forked from the initial commit and
deliberately excludes the 142 commits of adaptive development that exist on the
`demo` branch. The benchmark code under `wtb/` here is **unmodified original**.

This is your advantage: you are starting from clean code, and the DEV/TEST
protocol in section 2 can be honoured properly rather than retrofitted.

**Your incumbent on this branch is the clean baseline:**

| | task | session |
|---|---|---|
| `score_demo2_baseline` (this branch, unmodified code) | 354/1024 (34.6%) | 4/256 (1.56%) |

**For reference only** — the `demo` branch reached **404/1024 tasks and 12/256
sessions** through 17 scored runs of adaptive development against the full
benchmark. That number is contaminated development-set evidence and is **not**
your target; it is the bar a clean method should aim to beat *honestly*. Every
measurement quoted in sections 3, 4 and 8 of this document was obtained on that
branch, so treat those numbers as strong priors to re-verify on DEV, not as
facts about this branch.

**Offline forensics available on this branch** (section 8, Iteration 0):
`result_demo2_baseline/` holds a full 1024-task run with complete per-step
inference logs; `result_baseline_20/` and `result_strict/` hold smaller runs.
That is enough to build the analysis harness and rank opportunities without
spending any GPU.

You may consult the `demo` branch (`git show demo:<path>`) to read prior
implementations and avoid re-deriving dead ends — but do **not** merge it. The
value of this branch is that it is clean.

---

## 0. Environment, verified facts

Repo: `~/wildtoolbench_workspace_vllm`, branch `demo`.
Benchmark: `WildToolBench/wild-tool-bench/`.

- Data: `data/Wild-Tool-Bench.jsonl` — **256 sessions x 4 tasks = 1024 tasks**.
- Task types: Single-Tool 256, Clarify 256, Chat 256, Parallel Multi-Tool 156,
  Mixed Multi-Tool 84, Sequential Multi-Tool 16.
- **Mean 4.9 tools per session** (max 17). Tool-retrieval / tool-RAG methods
  from the literature are aimed at 50+ tool registries and are **not
  applicable here** — do not pursue them.
- Model handler: `wtb/model_handler/base_handler.py`, OpenAI-compatible client
  in `wtb/model_handler/api_inference/oai.py`. Served by local vLLM.
- Scoring: `wtb/eval_runner.py`, checker `wtb/checker_utils.py`.

**Scoring mechanics that matter (verified by reading the code — re-verify, do
not trust this blindly):**

- A session scores 1 only if **all 4 tasks** are correct.
- For task *N*, the model is fed a **gold-injected history** rebuilt from
  `english_answer_list[0..N-1]` — the ideal trajectory, not the model's own
  output. Tasks are therefore near-independent; failures do not cascade.
- `prepare_to_answer` and `ask_user_for_required_parameters` are **not callable
  tools** and are absent from the tool list. The model performs them by
  replying in plain text, and **the checker accepts ANY text** when gold is one
  of these. It matches the action type only.
- Argument keys must match the gold key set **exactly**; extra keys fail.
  Strings: exact -> whitespace/case-normalised -> edit ratio >= 0.8 (len < 10)
  -> ROUGE-L >= 0.7.
- Context is never a constraint: input tokens by turn are 706 / 1302 / 1770 /
  2263 (mean), max 5427, against a 32768 window. **Max 17% utilisation.**

**Current state of the art (see BRANCH CONTEXT above):**

| | task | session |
|---|---|---|
| Clean baseline on this branch | 354/1024 (34.6%) | 4/256 (1.56%) |
| Best on the contaminated `demo` branch | 404/1024 (39.5%) | 12/256 (4.7%) |

**Empirical session/task relationship.** Across all 17 scored runs, sessions
land **1.56x higher** than independence (`p^4`) predicts — tasks are correlated
within a session. Inverting that empirical multiplier:

- 6% sessions needs ~44% task accuracy
- 10% sessions needs ~50% task accuracy
- 15% sessions needs ~56% task accuracy

Current task accuracy is 39.5%. **A +4.5 point task gain is worth roughly
+1.3 points of session accuracy.** Use this to convert any measured task-level
delta into a session-level projection before committing GPU time.

---

## 1. Non-negotiable integrity rules

Violating any of these invalidates the result. Check your own work against them
before reporting any number.

1. **Never read `english_answer_list[i]` (or anything derived from it) for the
   task currently being answered.** Prior turns' gold is legitimately part of
   the injected history; the current turn's gold is the answer key.
2. **Never read** task ids, task-type labels, turn-subtype labels, the tool-call
   graph, `is_optimal`, or any scorer output at inference time.
3. **No benchmark-specific hardcoding**: no tool names, no field names, no
   session ids, no domain vocabulary, no thresholds hand-tuned against scores.
   Any constant you introduce must be justified from a general principle, not
   selected by scoring candidates on the eval set. Prefer parameter-free rules
   derived from the conversation's own content.
4. **Do not modify** `eval_runner.py`, `checker_utils.py`, `tool_call_graph.py`,
   or the dataset. You change how the model is *run*, never how it is *graded*.
5. Every run goes to a **fresh** `--result-dir`; generation silently skips ids
   that already have results.
6. After every run, confirm `total_count` is 1024 tasks / 256 sessions and grep
   the log for `Skipping errored result`. Timeouts silently shrink denominators
   and inflate accuracy.

---

## 2. Contamination — read this before designing the protocol

This repo has been adaptively developed against **all 256 sessions** across 17
scored runs. Any further number measured on the full set is
development-set evidence, not a generalisation result. An ICLR-calibre claim
requires fixing this.

**Mandatory protocol:**

- Deterministically split the 256 sessions into **DEV = 64** and
  **TEST = 192** (e.g. `id % 4 == 0` -> DEV). Write the split to a file and
  never change it.
- All exploration, tuning, and offline forensics happen on **DEV only**.
- **TEST is run at most twice in total across all 4 iterations** — ideally once,
  at the end. Report the TEST number as the headline result.
- Report the DEV/TEST gap honestly. A large gap is itself a finding.
- Use the existing `--run-ids` mechanism with
  `test_case_ids_to_generate.json` to restrict a run to a subset.

---

## 3. Dead ends — do not re-run these

These were measured on this exact model and benchmark. Re-deriving them wastes
your budget. If you believe one deserves revisiting, you must first state what
new evidence changes the conclusion.

| Idea | Measured outcome |
|---|---|
| Behavioural system prompts (triage / strict rules) | Task 34.4% -> 34.4% at full scale. A 20-session pilot dropped Qwen3-0.6B from 35.0% to 28.3%. Documented case: a "do not add unrequested params" rule made the model *drop* a parameter the task required. WTB's reviewers also object to system prompts on principle. |
| Self-consistency / majority voting over generations | Full 256-session run: **81 tasks fixed, 81 broken, net exactly 0.** Root cause: in **84% of steps all 6 samples were identical**, including 301 steps that were unanimously wrong. Errors are **systematic, not random**, so generative voting cannot help. |
| Memory / context digests (session ledger of facts) | Null. Context is at most 17% of the window and coreference accuracy is **depth-invariant** (33% / 33% / 31% at turns 1/2/3). This is not a memory problem. |
| Enriching surfaced facts with descriptive prose | Net **-6 Chat tasks**. All 18 broken Chat tasks flipped `text -> CALLED <tool>`: a dense structured block reads as a queryable database and induces tool calls on turns that need a plain answer. |
| Canonicalising string args against visible values | Fixes **2 of 94** string mismatches. |
| Recency preference for wrong-entity choices | 48% would fix, 50% would break. Coin flip. |
| Dropping optional keys whose key-concept is unmentioned | Correctly drops 36, wrongly drops 59: **net -23** (38% precision). |
| Wrapping text clarifications into `ask_user_for_required_parameters` | Fixes **zero** — plain text already scores correct 251 times when gold wants an ask. Would also emit a tool name absent from the registry. |

---

## 4. The measured failure landscape (starting point, re-verify on DEV)

Where sessions die (best recent run): **114 sessions die at turn 0**, 77 at
turn 1, 39 at turn 2, 17 at turn 3. **Turn 0 has the most session leverage and
is the least worked on** — it has no history, so it is pure comprehension of a
single user message against ~5 schemas.

Dominant remaining failure classes (whole run):

- `text instead of a tool call` (~109-125) — the single largest class.
- `guessed a value instead of asking` (~106).
- `wrong tool / wrong grouping / wrong count` (~101).
- `wrong argument value` (~93).
- `missing keys` (61: **51 the model never emitted**, 10 caused by an
  over-aggressive filter).
- `extra keys` (~60).

**Two findings worth building on — both are ours, neither is in any paper:**

1. **Ordinal reference cliff.** When the user turn (or their reply to a
   clarification) contains an ordinal/positional reference — "the first",
   "the last two", "the sixth holiday", "the one you mentioned in the first
   round" — accuracy **halves**:

   | user turn | accuracy |
   |---|---|
   | no ordinal reference | **43.7%** (345/789) |
   | contains an ordinal | **22.6%** (53/235) |

   42% of prior-turn observations return an ordered list of objects — the thing
   being pointed at. Positional indexing of surfaced facts was recently added
   but **has not yet been scored**. Measure its effect early.

2. **Wrong-entity selection dominates value errors.** Of 91 genuine string-argument
   failures: **47% are cases where BOTH the model's value and the gold value are
   already visible in the conversation** (`city: Chicago` when the dialogue had
   moved to Las Vegas; `ip: 228.121.22.88` when gold was `125.532.55.09`). Only
   8% are copy errors. These are **selection** failures, not formatting failures.

---

## 5. Literature you should build on (and one that should make you cautious)

Read these before designing. Cite them in your write-up.

- **WildToolBench** (ICLR 2026) — arXiv 2604.06185. Its own error taxonomy over
  57 LLMs finds "Wrong Name / Missing Info" and **"Redundant Call"** the most
  prevalent errors, and identifies a **cautious vs eager** dichotomy: models
  either over-refuse (Gemini-2.0-Thinking 24.6% refusal) or over-act (Grok-4
  24.1% wrong-name). It concludes the frontier is **higher-order planning and
  intent understanding**, not syntax. Note that specialised tool models
  (xLAM-2-70B, Watt-8B) have >30% wrong-name rates — "pseudo-capability".

- **The Constraint Tax** — arXiv 2605.26128, and **Constraint Tax in
  Open-Weight LLMs: Tool Calling Suppression Under Structured Output
  Constraints** — arXiv 2606.25605. **This is the single most important paper
  for you.** Hard schema-constrained decoding raised schema validity 61.5% ->
  100% but *lowered* answer accuracy 19.7% -> 11.0%. In a tool-call analogue,
  prompt-only JSON reached **91.5% executable accuracy while the same hard
  schema reached 48.0%** — both 100% schema-valid. The error becomes semantic,
  not structural. Their prescription is **"reason free, constrain late"**: use
  the least intrusive constraint that satisfies the contract, and package the
  answer only after the model has solved the task. **Therefore: do not naively
  wrap generation in `guided_json`.** If you use constrained decoding at all,
  apply it only to the final packaging step, and measure the tax explicitly.

- **XGrammar-2** — arXiv 2601.04426. The counterweight: grammar-constrained
  decoding took Llama-3.2-1B from 6.07% to 32.84% correct call rate. Reconcile
  this against the constraint tax — the difference is likely *what* is being
  constrained and *when*.

- **Uncertainty Decomposition for Clarification Seeking in LLM Agents** —
  arXiv 2606.19559; **Ask or Assume?** — arXiv 2603.26233; **Structured
  Uncertainty guided Clarification for LLM Agents** (ACL Findings 2026). The
  key idea: **decompose action confidence from request underspecification**.
  Deciding "ask vs act" from a single confidence signal conflates two different
  uncertainties. This maps directly onto WTB's Clarify category (256 tasks) and
  the cautious/eager paradox.

- **T1: Tool-integrated Verification for Test-time Compute Scaling in Small
  Language Models** — arXiv 2504.04718. Small models are unreliable *generative*
  verifiers but become reliable when verification is grounded in an external
  check. Supports moving work from generation to grounded discrimination.

- **Structured Output Collapses Answer Diversity Across 44 Language Models** —
  arXiv 2607.18476. Explains our 84%-unanimous finding: tool-call formatting
  itself collapses diversity, which is *why* generative self-consistency failed.

- **Selectivity collapse / over-tooled agents** — arXiv 2605.24660,
  2605.18857. Included so you can rule it out: WTB has ~5 tools per session,
  so this literature does not apply.

---

## 6. The central hypothesis to test

Everything above converges on one testable claim, and it is the strongest
candidate for a novel contribution:

> **Generative self-consistency fails on small-model tool use because the
> errors are systematic and structured decoding collapses diversity. But the
> model's *discriminative* competence is far higher than its *generative*
> competence, and its discriminative errors are not perfectly correlated with
> its generative ones. Therefore: keep generation free and unconstrained, and
> repair it with cheap, grounded, multiple-choice discrimination over an
> explicitly enumerated candidate set.**

This is "reason free, constrain late" pushed one step further: constrain not
the *syntax* but the *choice*.

Why this is novel and worth a paper:

- Nobody has tested discriminative self-verification on multi-turn tool use;
  the self-consistency literature is almost entirely generative.
- We have a strong measured negative result (voting = net zero, 84% unanimous)
  to contrast against — that is exactly the kind of paired result reviewers
  reward.
- The constraint-tax literature says syntax constraints hurt semantics. Our
  claim is that *semantic* constraints (choosing among visible entities) should
  not, because the candidate set is grounded in evidence rather than imposed.
- It explains, rather than merely patches, the dominant measured failure:
  47% of value errors are selection among visible candidates.

**Falsifier you must run:** if discriminative accuracy on a constructed
multiple-choice version of these decisions is no better than the model's
generative accuracy, the hypothesis is dead. Test this **offline in iteration 1
before implementing anything**.

---

## 7. Candidate mechanisms (implement the ones the data supports)

Do not implement all of these. Rank them by measured opportunity on DEV, then
build the top one or two.

**M1. Referential Slot Discrimination (targets the ordinal cliff + wrong-entity,
~235 + ~43 tasks).**
Enumerate the entity candidates visible in prior observations, with their
position and identifying fields. When a proposed argument value corresponds to
an enumerable slot, re-pose it as a multiple-choice question to the same model
("which of these is the user referring to?"), scored by logprob over the option
labels rather than free generation. Commit only if the discriminative margin
exceeds the model's own indifference. Two things make this cheap: a single
forward pass per slot, and vLLM can return logprobs so you may not need to
generate at all.

**M2. Two-signal ask-vs-act policy (targets ~106 guessed-instead-of-asking and
part of the 109+ text-instead-of-call).**
Following the uncertainty-decomposition literature, compute two *separate*
signals: (a) **action confidence** — is the tool choice stable? (b) **request
underspecification** — for each schema-required parameter, is its value
determined by the conversation? Crucially these should be assessed
*discriminatively* (per-parameter yes/no) rather than by asking the model to
introspect globally. Act when confidence is high and specification is complete;
ask when specification is incomplete; answer when no tool is licensed. Note the
scorer asymmetry: any text scores correct when gold wants an ask, so converting
a doomed call into a question is cheap — but a false conversion breaks a
passing task, so measure both directions.

**M3. Turn-0 specialisation (targets 114/256 session deaths, the largest single
lever, and the least explored).**
Turn 0 has no history and no ledger, so every history-based mechanism is inert
there. Diagnose turn-0 failures separately and design for them specifically.
This is where the highest session leverage sits and where nothing has been
tried.

**M4. Obligation completeness check (targets wrong grouping/count and WTB's own
"Redundant Call" finding).**
WTB's paper names Redundant Call as a top error across 57 models. Independently
check, discriminatively, (a) whether each proposed call is already satisfied by
a visible result, and (b) whether the user's message contains an additional
independent obligation that the proposal omits. Parallel-call completeness is
worth both task accuracy and Optimal Path Rate.

**M5. Reason-free / constrain-late packaging.**
If and only if M1-M4 leave structural errors on the table: let the model reason
and decide in free text, then package into JSON as a separate constrained step.
Measure the constraint tax explicitly by running both packaged and unpackaged
variants on DEV.

---

## 8. Iteration plan and budget

**Iteration 0 — offline forensics (NO GPU). Do this first and thoroughly.**

All existing runs are in `result_*/` and `score_*/` with **full per-step
inference logs**: the exact messages sent, the tools, the model's output, and
the gold candidates. Almost every question below can be answered offline for
free. This is how every finding in section 4 was obtained.

Deliverables:
- The DEV/TEST split file.
- A reusable offline analysis harness with, at minimum: per-turn session
  survival; failure taxonomy by class x turn; and a **counterfactual replay**
  that answers "if mechanism X had fired here, would the task have flipped?"
  scored against **actual task outcome**, not an intermediate proxy.
- A ranked table of opportunity sizes on DEV.

**Critical methodological warning, learned the hard way:** an earlier gate was
scored by "was this *step's action* correct" and looked strongly net-negative
(67 fix / 104 break). Re-scored against **final task outcome** the same
mechanism was net-positive (40 fix / 5 break) — because most of those steps
went on to fail on arguments anyway. **Always score a candidate mechanism
against the outcome you actually care about.** Also always measure what a
mechanism would do to currently-**passing** tasks, not only failing ones.

**Iteration 1 — falsify the central hypothesis, cheaply.**
Construct, from existing logs, multiple-choice versions of the decisions M1/M2
would make, where the gold answer is known. Measure the model's discriminative
accuracy on them with a small number of GPU calls (no full run). If
discrimination is not clearly better than generation, abandon the thesis and
fall back to M3/M4. Report the number either way.

**Iteration 2 — implement the winner, pilot on DEV (64 sessions).**
Implement behind an env-var flag, default off, so it can be disabled without a
revert. Unit-test the mechanism in isolation. Run DEV only. Compare against the
same DEV subset of the best existing run — do **not** compare a DEV number to a
full-set number.

**Iteration 3 — combine, ablate, and run TEST once.**
Add the second-ranked mechanism only if iteration 2 showed a real gain. Run the
held-out TEST set. Report: headline TEST session and task accuracy, per-mechanism
ablation on DEV, DEV/TEST gap, and the accompanying WTB metrics (per task type,
per turn, Optimal Path Rate, Accomplish Progress Rate) so that a gain is not
just a trade between categories.

**Stop rules.** If iteration 1 falsifies the hypothesis, do not implement M1/M2
— pivot immediately. If a DEV pilot shows a net change within +/-2 tasks on 64
sessions, treat it as noise and do not promote it to TEST. Never spend the last
iteration on a mechanism that has not shown a DEV gain.

---

## 9. What to report at the end

Write `RESEARCH_REPORT.md` containing:

1. **Headline**: TEST-set session and task accuracy vs the 12/256 and 404/1024
   incumbent, with the DEV/TEST gap stated plainly.
2. **The method**, described so it could be reimplemented from the text alone,
   with an explicit statement of what it does *not* use (no gold, no labels, no
   system prompt, no training).
3. **Ablations** on DEV for each component.
4. **Negative results**, including anything you falsified. The
   generative-vs-discriminative contrast is a contribution in its own right.
5. **Statistical honesty**: session counts are small. Report McNemar on the
   task level; state plainly when a session-level difference is not significant
   (a 4 -> 8 session change measured p ~ 0.22 earlier — directional, not proven).
6. **Threats to validity**: the contamination history of this repo, the single
   model studied, and any constants you could not derive from first principles.

---

## 10. Working style

- Measure before you build; build only what the measurement supports.
- Prefer mechanisms that change **what the model sees or chooses among** over
  mechanisms that **override its decisions** — the former have produced every
  gain in this repo so far, the latter have consistently measured net-zero.
- When a result contradicts your expectation, report it rather than tuning
  until it agrees.
- Keep every change behind a flag and keep the incumbent runnable.
- Commit with messages that record the measurement that justified the change,
  including the counter-evidence.

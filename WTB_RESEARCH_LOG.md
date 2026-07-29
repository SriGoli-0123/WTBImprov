# WildToolBench Improvement — Research Log & Handoff Document

Self-contained summary of diagnosis, experiments (including failures), and open
proposals. Written to be portable: another agent/IDE should be able to pick up
from here without re-deriving anything.

Model under study: **Qwen2.5-7B-Instruct** via vLLM.
Repo: `SriGoli-0123/WTBImprov`, branch `demo`.

---

## Current implementation: COG-FS (unscored)

The active `demo` method is now **Counterfactual Obligation Graph with Frontier
Synthesis (COG-FS)**. It replaces CAV's veto-oriented decision rule while
retaining its strict evaluator firewall. The ledger, TTM, consensus, hard-gate,
and CAV sections below remain as historical and negative-result evidence.

COG-FS preserves the original date-only system prompt and receives exactly
native `messages` plus the supplied `tools`. It rejects evaluator objects such
as answer lists, IDs, graph state, expected values, and scores.

It first generates an untouched anchor, then produces subtractive views that
remove the latest event, prior call syntax, or one general clause of the latest
user message. A call's disappearance identifies which user event or clause
caused it. Unlike CAV, calls suppressed in the anchor may be recovered from a
shadow, confirmed on the untouched conversation with their schema isolated, and
combined with other supported calls into the complete currently executable
frontier. A narrow projection path can also recover a single well-supported
tool obligation from a text anchor without exposing any benchmark category.

Before emission, every call is checked against the supplied schema and visible
evidence. Unsupported required values suppress a call; unknown keys and
unsupported optional values are removed; completed exact calls require a new
causal authorization. If intervention leaves no executable action, the same
model generates text from the unchanged conversation with tool use disabled.
No behavioral or corrective prompt is inserted.

The hard budget is eight generations per decision, served by one shared model:
one anchor plus no more than seven shadows, confirmations, projections, or
fallbacks. Identical views are deduplicated.

Implementation:

- `WildToolBench/wild-tool-bench/wtb/model_handler/cogfs.py`
- integration in `base_handler.py`
- text fallback in `api_inference/oai.py`
- fifteen unit tests in `tests/test_cogfs.py`

COG-FS has **not** been run or scored on WTB. The promotion target is to exceed
the recorded best of **12/256 sessions** and **404/1024 tasks** without merely
trading away task-type, accomplishment-progress, or optimal-path performance.

The immediately preceding CAV run was a negative result:

- task: **397/1024 (38.77%)**
- session: **6/256 (2.34%)**
- Single-Tool 94/256; Multi-Tool 41/256; Clarify 52/256; Chat 210/256
- accomplishment progress: 158/597 (26.47%)
- optimal path: 39/240 (16.25%)

This confirmed that stronger Chat performance alone is insufficient: session
accuracy requires recovering complete multi-action obligations while avoiding
regressions in tool execution.

Because this repository has already been adaptively developed against the full
benchmark, any next WTB run is exploratory/development-set evidence, not an
untouched generalization result. See `IMPROVEMENT_METHOD.md` for the exact
algorithm and firewall.

---

## 1. Benchmark mechanics (non-obvious, verified from code)

**Scoring.** A session = 4 tasks. `session_correct` requires **all 4** tasks correct.
Baseline: task 34.4%, session 1.6%. Because tasks are near-independent (see below),
`session ≈ task^4` (0.344^4 = 1.4% ≈ measured 1.6%).

**What the model actually receives per task** (built in
`wtb/model_handler/base_handler.py::_pre_messages_processing`):
- `system`: `Current Date: <english_env_info>` — that is the ENTIRE original system prompt.
- history messages, then the current user turn (`english_tasks[i]`)
- `tools` param = `english_tools` (mean 4.9 tools/session)

**Critical: history is GOLD-injected.** For task *N*, the history is reconstructed from
`english_answer_list[0..N-1]` — the *ideal* trajectory, not the model's own output.
Consequences:
- The model never sees its own earlier mistakes → **no error cascade between tasks**.
- Tasks are effectively independent exams → session ≈ task^4 holds.

**Fields that never reach the model:** `english_messages` (display-only transcript,
contains `"====="` separators). Don't waste time on it.

**Pseudo-actions.** `prepare_to_answer` and `ask_user_for_required_parameters` are NOT
callable tools and are not in the tool list. The model "performs" them by replying in
plain text. **The checker accepts ANY text when gold is one of these** — it matches only
the action type. This asymmetry is exploitable (see Gates below).

**Graph/ordering.** `english_answer_list[i][k]` has `dependency_list` + `idx`, which build
a DAG (`wtb/tool_call_graph.py`) defining legal step orderings. `is_optimal` = took the
minimum-depth path. Independent calls in one turn = optimal; one-at-a-time = legal but
non-optimal.

**Checker** (`wtb/checker_utils.py`): schema validation (type/enum/required/undefined),
then value match. Strings: exact → normalized → edit-ratio ≥0.8 (len<10) → ROUGE-L ≥0.7.
**Argument keys must match the gold key set exactly** — extra keys fail.

**Context size (measured).** Input tokens by turn: 706 / 1302 / 1770 / **2263** (mean),
max ever 5427. Window is 32768 → **max 17% utilization**. There is NO memory pressure.

---

## 2. Failure taxonomy (baseline, 672 failures of 1024 tasks)

| Class | Count | Share | Mechanically fixable? |
|---|---|---|---|
| Guessed a value instead of asking | 141 | 21% | Yes — provenance gate |
| Described the call instead of making it | 92 | 14% | No — judgment |
| Re-called a tool whose result was already present | 88 | 13% | Partly — duplicate gate |
| Extra hallucinated argument keys | 113 | 17% | **Yes — grammar lock** |
| Wrong *value* in correctly-shaped arg | 102 | 15% | No — knowledge/reasoning |
| Wrong tool entirely | 57 | 8% | Partly (12 are sibling confusions) |
| Right tools, wrong grouping/count | ~67 | 10% | No |
| Type/enum/nested schema violations | ~24 | 4% | **Yes — grammar lock** |

**Depth breakdown of gate coverage** (measured):

| | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| Gate-covered failures | 99 | 86 | 90 | 85 |
| Total failures | 141 | 164 | 178 | 189 |
| **Coverage** | 70% | 52% | 51% | **45%** |

Gates hold constant in absolute terms; failures GROW with depth, and all growth is in
uncovered classes: text-instead-of-call 5→32, wrong value 18→39, wrong tool 19→33.

**Referential load cliff:** 0 refs → 34.6%, 1–2 refs → 36.1%, **3+ refs → 17.9%** (n=56).

**Turn-subtype accuracy by depth** (key negative result):

| Subtype | T1 | T2 | T3 |
|---|---|---|---|
| Coreferential Reference | 33% | 33% | 31% |
| Partial Information | 41% | 30% | 31% |
| Long-Range Dependency | — | 26% | 22% |

Coreference is **depth-invariant** → information is being carried fine; the problem is
NOT retention.

---

## 3. Experiments run — including everything that FAILED

### 3.1 Prior work (Antigravity, 3 prompt-based "pre-execution check" methods)
All rewrote arguments of an already-emitted call. Result: task accuracy 34.4% → 25.3% /
33.9% / 23.6%. **Failed** — they act after the action-mode decision, so they cannot touch
the 445 action-name errors, and they corrupted already-correct args.

### 3.2 Strict-guideline system prompt (user's earlier 20-session pilot)
Qwen3-0.6B: 35.0% → 28.3%. Qwen2.5-7B: 36.25% → 35.0%. Sessions 0% in all cases.
Documented case: Rule "don't add unrequested params" made the model DROP
`includeStats: true` which the task required. **Conclusion: prompt rules shift errors
downstream ("whack-a-mole").** Consistent with reviewer feedback that system prompts
narrow the model's thinking. **Do not pursue behavioral prompting.**

### 3.3 Consensus decoding (self-consistency voting) — FULL SCALE, NULL RESULT
Setup: 6 candidates/step (1 anchor at temp 0 + 5 at temp 0.8 via vLLM `n=`), hierarchical
vote: response mode → tool-name multiset → per-arg key majority → per-arg value majority.
Plus triage system prompt + Session Ledger. Full 256-session run.

**Result: task 34.4% → 34.4%. Session 1.6% → 1.2%. 81 tasks fixed, 81 broken, net 0.**

Why it failed (the key finding):
- **84% of steps (1357/1624) had ALL 6 candidates produce the IDENTICAL answer**, including
  301 steps that were unanimously WRONG.
- Errors are **systematic, not random**. Voting only helps random scatter.
- On the 253 split steps, the vote overrode the anchor 32 times: 15 → correct, 17 → wrong.
  A coin flip.

**Conclusion: retire voting.** 6× compute for zero gain. This is a publishable negative
result (full-scale ablation showing tool-use errors in small models are systematic).

### 3.4 Session Ledger (terse fact digest injected at recency position) — NULL
≤12 telegraphic fact lines distilled per turn, injected immediately before the current
user message. Ran on all 687 history-bearing steps. Per-turn accuracy moved <±1pp at
every depth. **Failed because the facts were already present and already accessible**
(see coreference depth-invariance above).

### 3.5 Prose-constraint mining — KILLED BEFORE BUILDING
Observation: 437 parameters state machine-checkable rules only in free-text descriptions
("in the format 'YYYY-MM-DD'", quoted option lists) vs 249 with formal `enum` fields.
Hypothesis: enforce prose rules. **Tested against actual failures: only 5 violations.**
The model obeys stated formats; it writes a well-formatted but semantically WRONG date.
Dead end.

### 3.6 Infrastructure fixes (kept)
- `patch_vllm.py`: patches vLLM `hermes_tool_parser` for multiple/parallel tool calls +
  repairs truncated JSON (brace balancing + progressive trim-back for mid-token cuts).
  **Must re-run after any vLLM reinstall, then restart the server.**
- Client-side recovery/abstain in `base_handler.py`: salvages tool calls returned as
  text; unrecoverable attempts abstain rather than pollute a vote.
- `.env` is gitignored → `cp .env.example .env` once per machine.
- Scorer takes the UNDERSCORE model name (`Qwen_Qwen2.5-7B-Instruct`).
- Generation resumes/skips ids already in a result dir → **always use a fresh
  `--result-dir` per experiment**.

---

## 4. Historical proposal: three mechanical gates

Framing: aviation safety engineering. A session is a 4-leg flight; one crash = mission
zero. Aviation did NOT solve this by making pilots think better (= system prompts,
which we proved fail). It used **interlocks** that make specific mistakes impossible.

Also framed as classical planning (STRIPS): compute action **applicability**
mechanically from preconditions instead of letting the LLM judge it.

Every turn, the harness partitions the registry (zero extra LLM calls):
- **Runnable** — all required params have values traceable to the conversation
- **Blocked** — some required param has no source anywhere
- **Spent** — this exact call with these args already ran; result is in context

Model chooses freely; harness intervenes only on provable contradiction:

**Gate 1 — Grammar lock (SAFEST, BUILD FIRST).** vLLM constrained decoding
(`guided_json`) built from the schema WTB already supplies. Makes unspellable: non-schema
keys, wrong types, off-enum values, non-existent tool names, and truncation (EOS blocked
until JSON completes). Applied ONLY after the model freely chose to call a tool — text
responses untouched. **Depth-invariant by construction** (schema doesn't change with turn).
Targets ~137 failures.

**Gate 2 — Provenance / Blocked → ask.** If a required param value has no source,
convert the turn into a question. Exploits the checker asymmetry (any text scores correct
when gold is `ask_user_for_required_parameters`). Targets 141.
**⚠ MEASURED RISK: 12% of required-param values in CURRENTLY-CORRECT calls are not
literally quotable** (e.g. `country: 'CN'` when user said "China"; structured objects
assembled from scattered mentions; reformatted IPs). A naive gate would BREAK these.
→ Must be conservative: fire only when there is no fuzzy match, no derivable date, no
structural assembly. Accept lower recall for near-zero false-block rate.

**Gate 3 — Spent / duplicate → answer.** Block exact-duplicate calls; answer from the
existing result. Targets a subset of 88. Risk: legitimate repeat calls.

**Honest projection:** task ~46–50%, session ~4–6% (was 1.6%). Derived from the near-miss
analysis: of the 27 sessions failing by exactly ONE task, 15 have a blocking failure
inside gate coverage.

**Session distribution (baseline) — why near-misses matter:**
0/4 correct: 54 sessions | 1/4: 87 | 2/4: 84 | **3/4: 27** | 4/4: 4

---

## 5. Multi-turn / memory analysis

**WTB is NOT a memory problem.** Three proofs:
1. Max 17% context utilization (mean 6.9% at turn 3). Memory systems (Memora, Mem0,
   A-Mem, MemGPT) target ~26K-token conversations (LoCoMo) — 11× larger. Their headline
   win is token reduction (Memora: 98% fewer tokens); worthless at 7% utilization.
2. Gold-injected history → no self-error cascade (disables the mechanism in
   "LLMs Get Lost in Multi-Turn Conversation", arXiv 2505.06120).
3. Coreference accuracy is depth-invariant (33/33/31%) → information is carried fine.

**What actually degrades: distractor growth, not information loss.** More turns = more
legitimate candidate values → more chances to bind the wrong one. Retention fine,
*discrimination* degrades.

**Candidate for WTB:** Conversational Query Rewriting (CQR) — rewrite ONLY the current
user turn into a self-contained one, keeping full history intact (zero information-loss
risk, unlike history-replacing summarization). Justification is *decomposition* (fewer
simultaneous constraints), not self-correction. Literature: >60% of follow-ups have
unresolved coreferences. Expected +2 to +5 task points; still a judgment aid, so the
unanimity finding limits it.

**For >4 turns (future):** memory becomes real past the window. Design constraint: the
checker demands character-exact identifiers, so memory MUST be **extractive/lossless for
values** (append-only value store with source anchors — cf. TriMem's raw-segment anchors,
AtomMem). Compress prose only, never values. The provenance gate then reads the value
store instead of scanning the conversation — same gate logic, no redesign.

Refs: Memora (MSR, ICML 2026) · arXiv 2505.06120 · Mem0 arXiv 2504.19413 ·
A-Mem arXiv 2502.12110 · CHIQ arXiv 2406.05013 · AtomMem arXiv 2606.19847

---

## 6. Transfer learning assessment (tau2-bench → WTB)

**tau2-bench inventory (local, `tau2-bench/data/tau2/domains/`):** retail 114, airline 50,
telecom 2285, banking_knowledge 97 = **2546 tasks**. Format:
`user_scenario.instructions` (incl. `known_info` / **`unknown_info`**) +
`evaluation_criteria.actions` = gold `{name, arguments}` sequence.

**Compatibility:**
- ✅ Gold actions with exact arguments exist → convertible to SFT targets.
- ✅ `unknown_info` explicitly models missing information → trains **ask-when-missing**,
  our single largest failure class (141).
- ✅ ID-chaining workflows (`find_user_id_by_name_zip` → `get_order_details` →
  `get_product_details`) train **value propagation from tool results** → maps to WTB's
  Long-Range Dependency (worst subtype, 22%).
- ✅ Strict schemas → trains argument minimalism (113 extra-key failures).
- ❌ **No gold dialogues** — the user is LLM-simulated at runtime. Building SFT data
  requires running rollouts to synthesize transcripts. Non-trivial work.
- ❌ **Zero domain overlap** (retail/airline/telecom/banking vs WTB's diverse APIs).
  Transfer must be at the *skill* level, not domain level.
- ❌ Different reward shape (tau2 = final DB state + action set; WTB = exact arg match).
  Matters little for SFT.

**Literature on cross-benchmark transfer (modest):** fine-tuned models on BFCL/τ-bench
gain **0 to +5.0 points**; General Agent RL training transfers **+5.3pp to τ²-Bench**.

**KEY METHODOLOGICAL POINT:** WTB ships 1024 gold trajectories WITH full dialogues —
ideal in-distribution SFT data — BUT training on them is **test-set contamination**;
resulting WTB numbers would be invalid to report. **This is precisely why transfer
learning is the methodologically correct route**: out-of-distribution training data
leaves the WTB test set clean.

**Recommended data sources, best first:**
1. Large public function-calling SFT corpora (APIGen/xLAM ~60K verified trajectories,
   ToolBench, Glaive, Hermes-FC) — orders of magnitude more data than tau2, already
   dialogue-formatted. Best effort/reward ratio.
2. tau2-bench rollouts — smaller, needs synthesis work, but uniquely trains
   ask-when-missing and ID-chaining, our top two mechanical/judgment gaps.
3. WTB held-out split (e.g. train 56 sessions / test 200) — in-distribution and clean
   IF you report only on the held-out set. Reduces eval size but is defensible.

**Realistic expectation:** +2 to +5 task points from transfer alone → session ~2–3%.
Combined with gates (~46–50% task) it is the only credible path toward >15% session,
and even then >15% needs task accuracy ≈62%, which likely requires in-domain
(held-out-split) training, not transfer alone.

---

## 7. Configuration and commands

The variables below describe earlier experiments and are not part of active
COG-FS selection. COG-FS has one active runtime control:
`WTB_COGFS_WORKERS=8`. It is a hard per-decision generation budget and a
concurrency ceiling for one shared model, not a model-copy count.

Env knobs (in `wtb/model_handler/base_handler.py`):
| Var | Default | Meaning |
|---|---|---|
| `WTB_SC_N` | 5 | consensus candidates/step; **set to 1 — voting is retired** |
| `WTB_SC_TEMP` | 0.8 | diversity temperature |
| `WTB_SYS_MODE` | `triage` | `minimal` = original bare date prompt (**use `minimal`**) |
| `WTB_LEDGER` | 1 | `0` = off (**use 0** — measured null) |

Recommended COG-FS run:
```bash
python3 patch_vllm.py     # once per vLLM install, then restart server
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto \
  --enable-auto-tool-choice --tool-call-parser hermes &

cd WildToolBench/wild-tool-bench
cp .env.example .env
WTB_COGFS_WORKERS=8 \
python3 -u -m wtb.openfunctions_evaluation \
  --model Qwen/Qwen2.5-7B-Instruct --num-threads 8 --result-dir result_cogfs_fresh
python3 -u -m wtb.eval_runner \
  --model Qwen_Qwen2.5-7B-Instruct \
  --result-dir result_cogfs_fresh --score-dir score_cogfs_fresh
```

The result and score directory names must be new. WTB resumes/skips IDs already
present in an existing result directory.

---

## 8. Next evaluation steps

1. Run COG-FS once in a fresh result directory with the evaluator firewall
   unchanged.
2. Compare against the best recorded **12 sessions / 404 tasks**, not only the
   immediately preceding 6-session CAV run.
3. Report session and task totals plus every task type, layer, turn subtype,
   accomplishment-progress, and optimal-path metric.
4. Diff individual fixes and regressions, especially 3/4-session near misses,
   omitted multi-tool calls, false Chat-to-tool conversions, and duplicate
   actions.
5. Freeze COG-FS before evaluating on an independent messy-intent suite so the
   generality claim does not depend on repeated adaptation to WTB.

**Analysis principle that produced everything above:** measure the failure distribution
BEFORE proposing a fix, and test the proposed mechanism against currently-PASSING tasks
to quantify its false-positive risk. Three ideas (voting, ledger, prose-constraints) were
killed by this discipline — two after implementation, one before.

# Legacy Grounded experiment

> **Current method:** PRISM supersedes CONCORD, GAVEL-v2, and this wrapper. See
> [`method/PRISM.md`](PRISM.md) for the evaluator-free mechanism that resets
> policy inertia using only exact messages already present in the model input.
> CONCORD and GAVEL-v2 remain documented as ablations. This file is retained so
> the previous Grounded result remains reproducible.

# Deciding whether to act, before deciding how to act

## What the numbers said

Before writing anything, the baseline run (Qwen2.5-7B-Instruct, 34.4% of turns
correct, 4 of 256 sessions complete) was broken down by what kind of turn it
was and what went wrong.

| Turn type | correct | accuracy | wrong action | wrong arguments |
|---|---|---|---|---|
| Single-Tool | 98/256 | 38.3% | 61 | 97 |
| Chat (answer the user) | 177/256 | 69.1% | **79** | **0** |
| Clarify (ask the user) | 35/256 | 13.7% | **168** | 53 |
| Parallel Multi-Tool | 39/156 | 25.0% | 51 | 66 |
| Mixed Multi-Tool | 2/84 | 2.4% | 49 | 33 |
| Sequential Multi-Tool | 1/16 | 6.2% | 8 | 7 |

**416 of the 672 failed turns (62%) picked the wrong action.** Only the
remaining 256 filled a call in badly. That is the opposite of what you would
guess, and it decides what is worth building.

The two "special" actions -- asking the user for something missing, and
answering the user -- are not tools. There is nothing to call. A turn counts
as one of them when the model replies in plain words instead of emitting a
tool call. So most of the loss comes down to a single yes/no decision: **call
something, or say something.**

Splitting the failures by which way that decision went:

| Turn type | called ✓ | called ✗ | spoke ✓ | spoke ✗ |
|---|---|---|---|---|
| Single-Tool | 98 | 123 | 0 | 35 |
| Chat | 0 | **79** | **177** | **0** |
| Clarify | 0 | **125** | 35 | 96 |
| Parallel Multi-Tool | 37 | 103 | 2 | 14 |
| Mixed Multi-Tool | 2 | 79 | 0 | 3 |

On answer-the-user turns, speaking is right **177 times out of 177**. Every
one of the 79 losses is a tool call that should have been a sentence.

Two warnings come out of the same table, and they shaped the design:

- 55 turns spoke when they should have called. A blanket preference for
  speaking would give back much of what it gains.
- 96 clarify turns asked correctly and then failed anyway, on the follow-up
  call once the user supplied the missing detail. Routing correctly is
  necessary but only converts about a quarter of the time.

## The method

Five rules. Each one is a sentence.

1. **If the conversation already answers the user, reply in words.**
2. **If a required value is not in the conversation, ask for it in words.**
3. **If a call is needed, keep going until nothing asked for is left over.**
4. **Fix the shape of the call: real keys only, declared types, declared order.**
5. **Never call something whose result is already in the conversation.**

Rules 1 and 2 are the two halves of the yes/no decision, and they are where
the 416 wrong-action turns live. Rule 3 exists because multi-tool turns fail
by stopping early -- only 27% of the required steps get emitted. Rules 4 and 5
are bookkeeping that costs nothing.

### Why rule 3 repeats

The model writes down what the user asked for as a numbered list, then is
asked, with its own list and its own calls in front of it, whether anything is
still uncovered -- and that repeats until it says it is done. Asking once does
not work, because the model that missed the third item is the same model being
asked to notice the third item is missing. "Is anything left, given what you
already wrote?" is a much easier question than "what did you forget?", and
repeating it converges instead of taking one shot.

A turn with only one requested outcome skips the check entirely; a single
request cannot be half done.

### Why rule 2 asks for a quote

The check is not "does this value look right", which a model will always
answer yes to. It is "quote the words that give you this value, or write NOT
STATED". A quote can be checked. An opinion cannot. Anything that follows
directly from what was said -- a date from "this weekend", a country code from
a country name, a value taken from an earlier tool result -- counts as given,
because roughly one in eight correct calls uses a value that is derived rather
than copied, and treating those as missing would break working turns.

### Why this should keep paying off on a better model

The rules that last are the ones that **spot a problem in code and hand the
work back to the model**. Rules 2 and 3 do that: the wrapper decides that a
value has no source, or that an item has no call, and then the model resolves
it. A better model resolves it better, so the wrapper gets more useful rather
than less.

The rules that do not last are the ones that replace the model's judgment with
a fixed heuristic. Rule 5 is one of those, and half of rule 1 is too: a strong
model already answers the user correctly nearly every time, so that repair has
nothing left to do. They are a floor, not the engine.

This is worth stating plainly because it is checkable: run each rule on and
off, on a small model and a large one. Any rule whose effect disappears on the
larger model is a patch, and should be reported as one rather than folded into
the headline. The two rules aimed at asking-instead-of-guessing and at
finishing the job are the ones expected to survive, because those are still
unsolved at the frontier -- the best model in the paper gets 52.3% on ask-turns
and 40.2% on multi-tool turns.

Nothing here is tuned to a particular model, and nothing is trained on this
benchmark.

## Running it

First start the model server in its own terminal and leave it running:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto \
    --enable-auto-tool-choice --tool-call-parser hermes
```

`.env` must point at it (`OPENAI_BASE_URL=http://localhost:8000/v1`).

In a second terminal, pick the ten sessions to work on, then run both arms.
The wrapper is off unless you ask for it, so the same command produces both
numbers:

```bash
python3 method/make_batches.py --batch 1     # writes the ten ids the runner reads

# baseline arm
python3 -u -m wtb.openfunctions_evaluation --model Qwen/Qwen2.5-7B-Instruct \
    --num-threads 8 --result-dir result_b1_baseline --run-ids
python3 -u -m wtb.eval_runner --model Qwen_Qwen2.5-7B-Instruct \
    --result-dir result_b1_baseline --score-dir score_b1_baseline

# method arm -- identical, one environment variable
WTB_METHOD=grounded python3 -u -m wtb.openfunctions_evaluation \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-threads 8 --result-dir result_b1_grounded --run-ids
python3 -u -m wtb.eval_runner --model Qwen_Qwen2.5-7B-Instruct \
    --result-dir result_b1_grounded --score-dir score_b1_grounded
```

Run the modules with `-m`; running the files directly breaks their imports.
Use a fresh `--result-dir` per arm per batch, or old results are silently
reused and the comparison is meaningless.

Batches:

```bash
python3 method/make_batches.py            # once: 40 dev sessions, 4 batches
python3 method/make_batches.py --batch 2  # move on once batch 1 is understood
```

## Running the whole benchmark

Leaving `--run-ids` off runs every session in the file, which is all 256.
There is nothing else to switch on.

One command does both arms and prints the comparison at the end:

```bash
bash method/run_full.sh
```

It checks the server is up and that the method arm really resolves to the
wrapper before it starts, writes to `result_full_baseline` /
`result_full_grounded` and `score_full_baseline` / `score_full_grounded`, and
keeps the logs in `logs_full/`. Useful variants:

```bash
bash method/run_full.sh --arm grounded    # only the method arm
bash method/run_full.sh --threads 16      # if the server can take it
bash method/run_full.sh --tag v2          # keep a second run separate
bash method/run_full.sh --fresh           # discard old results and start over
```

The same thing written out by hand, if you would rather see every step:

```bash
# plain arm
python3 -u -m wtb.openfunctions_evaluation --model Qwen/Qwen2.5-7B-Instruct \
    --num-threads 8 --result-dir result_full_baseline
python3 -u -m wtb.eval_runner --model Qwen_Qwen2.5-7B-Instruct \
    --result-dir result_full_baseline --score-dir score_full_baseline

# method arm -- same command, one environment variable
WTB_METHOD=grounded python3 -u -m wtb.openfunctions_evaluation \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-threads 8 --result-dir result_full_grounded
python3 -u -m wtb.eval_runner --model Qwen_Qwen2.5-7B-Instruct \
    --result-dir result_full_grounded --score-dir score_full_grounded

python3 method/compare.py score_full_baseline/Qwen_Qwen2.5-7B-Instruct \
                          score_full_grounded/Qwen_Qwen2.5-7B-Instruct
```

Two things worth knowing before starting a full run.

**It resumes.** Sessions that already have a result are skipped, so if the run
dies partway through, re-running the identical command carries on from where it
stopped. That also means a result directory must never be shared between two
different arms, or you get a silent mixture of both. Use `--fresh` (or
`--allow-overwrite` by hand) to start a directory over.

**It costs more than the plain run.** A turn answered in words costs one model
call and is returned untouched. A turn that needs a call costs about four, and
a multi-tool turn being repaired can reach eight. Over 256 sessions that is
roughly five thousand generations against about a thousand for the plain arm.
If that is too slow, both limits can be turned down from the shell:

```bash
WTB_MAX_CHECKS=1 WTB_COVERAGE_ROUNDS=1 WTB_METHOD=grounded ...
```

`WTB_COVERAGE_ROUNDS=1` gives back the single-shot version of rule 3, which is
the thing rule 3 exists to replace, so compare it against 3 rather than
reporting it as the method.

### If turns hang

Every extra question the wrapper asks has a short answer -- one word, a few
quoted phrases, a numbered list -- so all of them are capped. Without a cap a
single check can keep generating until the server's own limit, which is what
turns a handful of sessions into a stall.

Beyond that, repair work is optional by construction. Each turn gets a budget;
once it is spent, or if a check times out or fails for any other reason, the
model's own first draft is returned unchanged and that turn scores exactly as
the baseline would have. A missed repair costs one turn. A crash costs the run.
The line `[grounded] repair skipped, keeping the draft` in the log is that
happening, and it is worth counting -- if it appears often, the run is quietly
turning into the baseline.

The free parts of the method (real parameter names, declared types and order,
dropping a call whose answer is already in the conversation) cost no model call
and still apply even when the budget is gone.

| variable | default | what it does |
|---|---|---|
| `WTB_CHECK_TOKENS` | 256 | length cap on each check |
| `WTB_SPEAK_TOKENS` | 512 | length cap on the reply the wrapper asks for |
| `WTB_TURN_BUDGET` | 180 | seconds of repair allowed per turn |
| `WTB_REQUEST_TIMEOUT` | 600 | client timeout, method arm only |

Raising the timeout alone is not a fix. If one request genuinely needs ten
minutes, something is unbounded rather than slow, and the caps and the budget
are what deal with that. Note also that the run resumes: sessions that timed
out are the only ones missing, so re-running the same command retries just
those.

If the same session ids keep failing, it is not a tuning problem -- take the
ids from the log and look at them directly.

**And one thing about the number it gives you.** The first full run is clean:
nothing has been fitted to those sessions. It stops being clean the moment
something gets changed in response to what it says and it is run again. If
that happens, the honest thing is to report the first number, or to go back to
the dev batches and keep the rest sealed until the method has stopped moving.

Reading the result:

```bash
python3 method/compare.py score_b1_baseline/<model> score_b1_grounded/<model>
```

`compare.py` reports how many turns were fixed and how many were broken,
because on forty sessions the headline accuracy moves by about four points
from noise alone and a net figure hides the damage. A rule is worth keeping
when it fixes at least eight more turns than it breaks and breaks no more than
five.

## What to expect

Simulating repairs against the real session structure, rather than assuming
turns are independent:

| Repair | complete sessions |
|---|---|
| baseline | 4 |
| answer-the-user routing only | ~6 |
| \+ clarify routing | ~16 |
| \+ single-tool | ~22 |
| \+ multi-tool | ~33 |
| all of it, working well | ~56 |

Reaching 50 sessions needs roughly 68% of turns correct. No model in the
original paper exceeds 61%. That is the honest price of the target, and it is
only plausible because most of what is being fixed is mechanical rather than a
limit on what the model knows.

One caution about that table. Sessions do better than you would expect from
multiplying the per-type numbers together, because a session that goes well
tends to go well throughout. But that bonus shrinks as a model gets better:
it is 2.63x here, 1.84x for a tuned 7B, 1.51x for GPT-4o and 1.28x for the
best model in the paper. If this wrapper makes the model behave like a
stronger one, the bonus should fall with it. Planning on it staying high is
double-counting. The number to plan for is nearer 40 sessions than 56, and
the claim should rest on the per-type accuracies, which are measured, rather
than on a multiplier that moves in the wrong direction as things improve.

The last two batches are for confirming, not for tuning. The sealed sessions
get run once, at the end.

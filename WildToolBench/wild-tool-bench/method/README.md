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

The wrapper is off unless you ask for it, so the same command produces both
numbers:

```bash
# baseline
python wtb/_llm_response_generation.py --model Qwen/Qwen2.5-7B-Instruct \
    --result-dir result_baseline --run-ids
python wtb/eval_runner.py --model Qwen/Qwen2.5-7B-Instruct \
    --result-dir result_baseline --score-dir score_baseline

# with the method
WTB_METHOD=grounded python wtb/_llm_response_generation.py \
    --model Qwen/Qwen2.5-7B-Instruct --result-dir result_grounded --run-ids
python wtb/eval_runner.py --model Qwen/Qwen2.5-7B-Instruct \
    --result-dir result_grounded --score-dir score_grounded
```

Working in batches of ten:

```bash
python method/make_batches.py            # once: 40 dev sessions, 4 batches
python method/make_batches.py --batch 1  # load batch 1, then run with --run-ids
```

Reading the result:

```bash
python method/compare.py score_baseline/<model> score_grounded/<model>
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

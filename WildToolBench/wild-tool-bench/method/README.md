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
3. **If a call is needed, check the whole request is covered.**
4. **Fix the shape of the call: real keys only, declared types, declared order.**
5. **Never call something whose result is already in the conversation.**

Rules 1 and 2 are the two halves of the yes/no decision, and they are where
the 416 wrong-action turns live. Rule 3 exists because multi-tool turns fail
by stopping early -- only 27% of the required steps get emitted. Rules 4 and 5
are bookkeeping that costs nothing.

### Why rule 2 asks for a quote

The check is not "does this value look right", which a model will always
answer yes to. It is "quote the words that give you this value, or write NOT
STATED". A quote can be checked. An opinion cannot. Anything that follows
directly from what was said -- a date from "this weekend", a country code from
a country name, a value taken from an earlier tool result -- counts as given,
because roughly one in eight correct calls uses a value that is derived rather
than copied, and treating those as missing would break working turns.

### Why this should keep paying off on a better model

Rules 1, 2 and 3 ask the model questions. A stronger model answers them
better, so the wrapper gets more accurate rather than less useful. Rules 4 and
5 are fixed repairs that a stronger model will need less often; they are a
floor, not the engine. Nothing here is tuned to a particular model, and
nothing is trained on this benchmark.

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

The last two batches are for confirming, not for tuning. The sealed sessions
get run once, at the end.

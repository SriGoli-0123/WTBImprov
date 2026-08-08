"""Compare two runs turn by turn.

Accuracy on a forty-session split moves around by about four points from noise
alone, so a difference in the headline number tells you very little. What does
tell you something is which individual turns changed direction:

    fixed  -- wrong before, right now
    broke  -- right before, wrong now

A rule is worth keeping when it fixes a lot more than it breaks. Breaking
turns is expensive out of proportion to fixing them, because a session needs
every one of its turns to be right.

    python method/compare.py score/baseline score/grounded
"""

import collections
import json
import os
import sys


def load(score_dir):
    path = None
    for candidate in os.listdir(score_dir):
        if candidate.endswith("_score.jsonl"):
            path = os.path.join(score_dir, candidate)
    if path is None:
        raise SystemExit(f"no *_score.jsonl in {score_dir}")
    out = {}
    for line in open(path):
        row = json.loads(line)
        for i, result in enumerate(row["results"]):
            out[(row["id"], i)] = result["label"] == "correct"
    return out


def task_types(data_path):
    types = {}
    for line in open(data_path):
        row = json.loads(line)
        for i, name in enumerate(row["english_task_types"]):
            types[(row["id"], i)] = name
    return types


def main(before_dir, after_dir, data_path):
    before, after = load(before_dir), load(after_dir)
    shared = sorted(set(before) & set(after))
    if not shared:
        raise SystemExit("the two runs have no turns in common")
    types = task_types(data_path) if os.path.exists(data_path) else {}

    fixed = [k for k in shared if not before[k] and after[k]]
    broke = [k for k in shared if before[k] and not after[k]]

    print(f"turns compared: {len(shared)}")
    print(f"fixed: {len(fixed)}")
    print(f"broke: {len(broke)}")
    print(f"net:   {len(fixed) - len(broke):+d}")

    if types:
        per = collections.defaultdict(lambda: [0, 0])
        for k in fixed:
            per[types.get(k, "?")][0] += 1
        for k in broke:
            per[types.get(k, "?")][1] += 1
        print()
        print(f"{'type':24} {'fixed':>6} {'broke':>6} {'net':>6}")
        for name, (f, b) in sorted(per.items(), key=lambda x: -(x[1][0] - x[1][1])):
            print(f"{name:24} {f:6} {b:6} {f - b:+6d}")

    # sessions are what actually counts
    def sessions(d):
        by = collections.defaultdict(list)
        for (sid, _), ok in d.items():
            by[sid].append(ok)
        return sum(1 for v in by.values() if all(v)), len(by)

    b_ok, b_n = sessions({k: before[k] for k in shared})
    a_ok, a_n = sessions({k: after[k] for k in shared})
    print()
    print(f"complete sessions: {b_ok}/{b_n} -> {a_ok}/{a_n}  ({a_ok - b_ok:+d})")

    # on a small split the session count is often zero either way, so also
    # report how far the average session is from being complete
    def distance(d):
        by = collections.defaultdict(list)
        for (sid, _), ok in d.items():
            by[sid].append(ok)
        return sum(sum(1 for x in v if not x) for v in by.values()) / len(by)

    print(f"average turns wrong per session: "
          f"{distance({k: before[k] for k in shared}):.2f} -> "
          f"{distance({k: after[k] for k in shared}):.2f}")

    verdict = "KEEP" if (len(fixed) - len(broke) >= 8 and len(broke) <= 5) else (
        "KILL" if len(broke) >= len(fixed) or len(broke) >= 8 else "INCONCLUSIVE")
    print()
    print(f"verdict: {verdict}   (keep needs net >= +8 and broke <= 5)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(os.path.dirname(here), "data", "Wild-Tool-Bench.jsonl")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else default_data)

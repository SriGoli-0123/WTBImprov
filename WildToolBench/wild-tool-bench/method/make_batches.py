"""Carve a development split and write it out in batches of ten.

Everything outside the development split stays untouched until the method is
frozen. The split is written to disk so it can be checked, and the seed is
fixed so it can be reproduced.

    python method/make_batches.py            # create split + batches
    python method/make_batches.py --batch 1  # load batch 1 into the runner
"""

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "Wild-Tool-Bench.jsonl")
SPLIT = os.path.join(HERE, "dev_split.json")
RUNNER_IDS = os.path.join(ROOT, "wtb", "test_case_ids_to_generate.json")

DEV_SIZE = 40
BATCH = 10
SEED = 20240711


def build():
    ids = [json.loads(line)["id"] for line in open(DATA)]
    rng = random.Random(SEED)
    shuffled = sorted(ids)
    rng.shuffle(shuffled)
    dev, sealed = shuffled[:DEV_SIZE], shuffled[DEV_SIZE:]
    payload = {
        "seed": SEED,
        "dev": dev,
        "sealed_count": len(sealed),
        "batches": {str(i + 1): dev[i * BATCH:(i + 1) * BATCH] for i in range(DEV_SIZE // BATCH)},
    }
    with open(SPLIT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"development split: {len(dev)} sessions -> {SPLIT}")
    print(f"sealed:            {len(sealed)} sessions (do not run until frozen)")
    for name, batch in payload["batches"].items():
        print(f"  batch {name}: {len(batch)} sessions")
    return payload


def load_batch(n):
    payload = json.load(open(SPLIT))
    key = str(n)
    if key == "dev":
        ids = payload["dev"]
    elif key in payload["batches"]:
        ids = payload["batches"][key]
    else:
        raise SystemExit(f"no batch {n}; have {list(payload['batches'])} or 'dev'")
    with open(RUNNER_IDS, "w") as fh:
        json.dump(ids, fh, indent=2)
    print(f"loaded {len(ids)} session ids into {RUNNER_IDS}")
    print("run with --run-ids to restrict the run to these sessions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=None, help="batch number, or 'dev' for all 40")
    args = parser.parse_args()
    if args.batch:
        load_batch(args.batch)
    else:
        build()

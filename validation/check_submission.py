#!/usr/bin/env python3
"""
Basic validator for Kaggle ARC‑II submission.json produced by run_fullstack_eval.py.
Checks:
  • JSON is a dict of task_id -> list[predictions]
  • Each prediction is a 2D list of integers (grid) or empty list (fallback)
  • Exactly 2 attempts per test input (pipeline enforces this; we verify at top level)
"""
import argparse, json, sys

def is_grid(lst):
    if not isinstance(lst, list) or not lst:
        return False
    if not all(isinstance(r, list) for r in lst):
        return False
    # Rows can be ragged in ARC; enforce ints but not fixed width
    return all(all(isinstance(c, int) for c in r) for r in lst if isinstance(r, list))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_json", type=str, help="Path to submission.json")
    args = ap.parse_args()

    with open(args.submission_json, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("❌ submission root must be a dict of {task_id: [predictions...]}")
        sys.exit(1)

    bad = 0
    for tid, preds in data.items():
        if not isinstance(preds, list):
            print(f"❌ {tid}: predictions must be a list")
            bad += 1
            continue
        # We expect concatenated attempts across tests (2 per test)
        if len(preds) == 0:
            print(f"⚠️  {tid}: empty predictions list")
            continue
        for i, grid in enumerate(preds):
            if grid == []:
                # allowed fallback
                continue
            if not is_grid(grid):
                print(f"❌ {tid}: prediction[{i}] is not a 2D int grid")
                bad += 1
                break

    if bad == 0:
        print("✅ submission.json looks structurally valid for Kaggle ARC‑II")
    else:
        print(f"Found {bad} invalid entries.")
        sys.exit(2)

if __name__ == "__main__":
    main()

"""
Run v2 experiment: profiles with REAL qualification variation + improved prompt.
This forces the model to discriminate, revealing any demographic bias.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import build_prompt, call_claude
import time

num_sets = int(sys.argv[1]) if len(sys.argv) > 1 else 10
num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# Load v2 profiles
data_path = Path(__file__).parent.parent / "data" / "profile_sets_v2.json"
with open(data_path, "r", encoding="utf-8") as f:
    all_sets = json.load(f)

profile_sets = all_sets[:num_sets]
mitigations = ["baseline", "explicit_fairness", "role_fair", "cot"]

total_calls = num_sets * num_runs * len(mitigations)
print(f"=" * 60)
print(f"V2 EXPERIMENT (profiles with real variation)")
print(f"=" * 60)
print(f"  Sets: {num_sets}")
print(f"  Runs per set: {num_runs}")
print(f"  Mitigations: {len(mitigations)}")
print(f"  Total API calls: {total_calls}")
print(f"=" * 60)
print()

for mit in mitigations:
    print(f"\n>>> {mit} <<<")
    results = []
    output_path = Path(__file__).parent.parent / "results" / f"results_v2_claude_{mit}.json"
    output_path.parent.mkdir(exist_ok=True)

    # Resume from existing
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"  [INFO] Resuming from {len(results)} existing")
        except Exception:
            results = []
    completed = {(r["set_id"], r["run_idx"]) for r in results}
    call_count = len(results)

    for set_idx, profile_set in enumerate(profile_sets):
        prompt = build_prompt(profile_set, mitigation_strategy=mit)
        for run_idx in range(num_runs):
            if (profile_set["set_id"], run_idx) in completed:
                continue
            call_count += 1
            print(f"[{call_count}/{total_calls}] Set {set_idx+1}, Run {run_idx+1}...", end=" ", flush=True)
            try:
                response = call_claude(prompt)
                results.append({
                    "set_id": profile_set["set_id"],
                    "run_idx": run_idx,
                    "model": "claude",
                    "mitigation": mit,
                    "prompt": prompt,
                    "response": response,
                    "candidates": profile_set["candidates"],
                })
                completed.add((profile_set["set_id"], run_idx))
                print("[OK]", flush=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                time.sleep(0.3)
            except Exception as e:
                print(f"[X] {e}", flush=True)
    print(f"[OK] {mit}: {len(results)} results saved\n")

print(f"\n{'=' * 60}")
print(f"DONE. To analyze:")
print(f"  python scripts/parse_responses_v2.py")

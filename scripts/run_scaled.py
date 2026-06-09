"""
Run the experiment at a chosen scale.
Usage: python scripts/run_scaled.py [num_sets] [num_runs]
Defaults: 10 sets, 2 runs
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import run_experiment

num_sets = int(sys.argv[1]) if len(sys.argv) > 1 else 10
num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# Load profiles
data_path = Path(__file__).parent.parent / "data" / "profile_sets.json"
with open(data_path, "r", encoding="utf-8") as f:
    all_sets = json.load(f)

# Take first N
profile_sets = all_sets[:num_sets]
mitigations = ["baseline", "explicit_fairness", "role_fair", "cot"]

total_calls = num_sets * num_runs * len(mitigations)
print(f"=" * 60)
print(f"SCALED RUN")
print(f"=" * 60)
print(f"  Sets: {num_sets}")
print(f"  Runs per set: {num_runs}")
print(f"  Mitigations: {len(mitigations)}")
print(f"  Total API calls: {total_calls}")
print(f"  Est. time: ~{total_calls * 4 // 60} min")
print(f"=" * 60)
print()

for mit in mitigations:
    print(f"\n>>> {mit} <<<")
    results = run_experiment(
        profile_sets=profile_sets,
        model="claude",
        mitigation=mit,
        num_runs=num_runs,
    )
    print(f"[OK] {mit}: {len(results)} results\n")

print(f"\n{'=' * 60}")
print(f"DONE. Run: python scripts/parse_responses.py")

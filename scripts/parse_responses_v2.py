"""
Parse v2 results (separate output file to not overwrite v1).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_responses import parse_all_results

results_dir = Path(__file__).parent.parent / "results"
all_results = []
for mit in ["baseline", "explicit_fairness", "role_fair", "cot"]:
    path = results_dir / f"results_v2_claude_{mit}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            mit_results = json.load(f)
        all_results.extend(mit_results)
        print(f"  [OK] Loaded {len(mit_results)} from {path.name}")
    else:
        print(f"  [WARN] Missing: {path.name}")

if not all_results:
    print("[WARN] No v2 results found. Run scripts/run_v2.py first!")
else:
    print(f"\nLoaded {len(all_results)} total v2 results\n")
    parsed = parse_all_results(all_results)

    output_path = results_dir / "parsed_results_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved to: {output_path}")

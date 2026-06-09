"""
Pilot run: 2 profile sets, 1 run each, real API.
Tests that the full pipeline works end-to-end with the custom endpoint.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import call_claude, build_prompt
from generate_profiles import generate_all_sets

print("=" * 60)
print("PILOT RUN - Real API via custom endpoint")
print("=" * 60)

# Load profiles
data_path = Path(__file__).parent.parent / "data" / "profile_sets.json"
with open(data_path, "r", encoding="utf-8") as f:
    profile_sets = json.load(f)

# Take just 2 sets for pilot
pilot_sets = profile_sets[:2]

print(f"\nLoaded {len(pilot_sets)} profile sets for pilot")
print(f"Each set has {len(pilot_sets[0]['candidates'])} candidates\n")

results = []

for set_idx, profile_set in enumerate(pilot_sets):
    prompt = build_prompt(profile_set, mitigation_strategy="baseline")
    print(f"[Set {set_idx + 1}/{len(pilot_sets)}] Sending prompt ({len(prompt)} chars)...")

    try:
        response = call_claude(prompt)
        print(f"  [OK] Got response ({len(response)} chars)")

        # Try to extract JSON
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                print(f"  [OK] Parsed JSON: keys = {list(parsed.keys())}")
                if "scores" in parsed:
                    print(f"  [OK] Scores: {parsed['scores']}")
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse failed: {e}")
        else:
            print(f"  [WARN] No JSON block found in response")
            print(f"  First 200 chars: {response[:200]}")

        results.append({
            "set_id": profile_set["set_id"],
            "response": response,
            "candidates": profile_set["candidates"],
        })

    except Exception as e:
        print(f"  [X] Error: {e}")
        continue

print(f"\n{'=' * 60}")
print(f"PILOT COMPLETE: {len(results)}/{len(pilot_sets)} successful")
print(f"{'=' * 60}")

if len(results) == len(pilot_sets):
    print("\n[OK] All pilot runs succeeded. You can scale up to the full experiment.")
    print("\nTo run the full experiment, use:")
    print("  python scripts/run_experiment.py")
else:
    print("\n[WARN] Some runs failed. Check the errors above.")

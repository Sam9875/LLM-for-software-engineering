"""
Parse LLM responses to extract structured data (ranking, scores, justifications).
Handles malformed JSON and edge cases.
"""

import json
import re
from pathlib import Path


def extract_json_from_response(response_text):
    """
    Extract JSON from LLM response.
    Handles cases where JSON is wrapped in markdown code blocks.
    """
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON within markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the text
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def normalize_candidate_label(label):
    """
    Normalize candidate labels to 'Candidate N' format.
    Handles variations like 'candidate 1', 'Candidate One', '1', etc.
    """
    if not isinstance(label, str):
        return label

    label = label.strip()

    # Already in correct format
    if re.match(r"^Candidate\s+\d+$", label, re.IGNORECASE):
        return label

    # Extract number from various formats
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    # Try "Candidate X" pattern
    match = re.search(r"candidate\s+(\d+)", label, re.IGNORECASE)
    if match:
        return f"Candidate {match.group(1)}"

    # Try word numbers
    for word, num in number_words.items():
        if word in label.lower():
            return f"Candidate {num}"

    # Try just a number
    match = re.search(r"\d+", label)
    if match:
        return f"Candidate {match.group(0)}"

    return label


def parse_response(result):
    """
    Parse a single experiment result.

    Returns:
        dict with keys: set_id, run_idx, ranking (list), scores (dict),
                        justifications (dict), parse_success (bool)
    """
    response_text = result["response"]

    parsed = extract_json_from_response(response_text)

    if parsed is None:
        return {
            "set_id": result["set_id"],
            "run_idx": result["run_idx"],
            "model": result["model"],
            "mitigation": result["mitigation"],
            "ranking": None,
            "scores": None,
            "justifications": None,
            "parse_success": False,
            "error": "Failed to extract JSON from response",
        }

    # Extract and normalize ranking
    ranking = parsed.get("ranking", [])
    if isinstance(ranking, list):
        ranking = [normalize_candidate_label(r) for r in ranking]

    # Extract scores
    scores = parsed.get("scores", {})
    if isinstance(scores, dict):
        normalized_scores = {}
        for label, score in scores.items():
            normalized_label = normalize_candidate_label(label)
            try:
                normalized_scores[normalized_label] = float(score)
            except (ValueError, TypeError):
                continue
        scores = normalized_scores

    # Extract justifications
    justifications = parsed.get("justifications", {})
    if isinstance(justifications, dict):
        normalized_justifications = {
            normalize_candidate_label(k): v
            for k, v in justifications.items()
        }
        justifications = normalized_justifications

    return {
        "set_id": result["set_id"],
        "run_idx": result["run_idx"],
        "model": result["model"],
        "mitigation": result["mitigation"],
        "ranking": ranking,
        "scores": scores,
        "justifications": justifications,
        "parse_success": True,
        "candidates": result.get("candidates", []),
    }


def parse_all_results(results):
    """Parse all results and return structured data."""
    parsed_results = []
    success_count = 0

    for result in results:
        parsed = parse_response(result)
        parsed_results.append(parsed)
        if parsed["parse_success"]:
            success_count += 1

    print(f"[OK] Parsed {success_count}/{len(results)} results successfully")
    if success_count < len(results):
        print(f"[WARN] {len(results) - success_count} responses failed to parse")

    return parsed_results


def link_scores_to_demographics(parsed_result, profile_set):
    """
    Link scores/rankings to demographic groups.

    Returns:
        list of dicts with: candidate_label, score, rank, demographic_group, gender, ethnicity
    """
    candidates = profile_set["candidates"]
    scores = parsed_result.get("scores", {})
    ranking = parsed_result.get("ranking", [])

    # Create rank lookup
    rank_lookup = {label: idx + 1 for idx, label in enumerate(ranking)}

    linked_data = []
    for idx, candidate in enumerate(candidates):
        candidate_label = f"Candidate {idx + 1}"
        linked_data.append({
            "candidate_label": candidate_label,
            "set_id": parsed_result["set_id"],
            "run_idx": parsed_result["run_idx"],
            "model": parsed_result["model"],
            "mitigation": parsed_result["mitigation"],
            "demographic_group": candidate.get("demographic_group", "unknown"),
            "gender": candidate.get("gender", "unknown"),
            "ethnicity": candidate.get("ethnicity", "unknown"),
            "nationality": candidate.get("nationality", "unknown"),
            "score": scores.get(candidate_label),
            "rank": rank_lookup.get(candidate_label),
            "justification": parsed_result.get("justifications", {}).get(candidate_label, ""),
        })

    return linked_data


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load all mitigation result files
    results_dir = Path(__file__).parent.parent / "results"
    all_results = []
    for mit in ["baseline", "explicit_fairness", "role_fair", "cot"]:
        path = results_dir / f"results_claude_{mit}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                mit_results = json.load(f)
            all_results.extend(mit_results)
            print(f"  [OK] Loaded {len(mit_results)} from {path.name}")
        else:
            print(f"  [WARN] Missing: {path.name}")

    if not all_results:
        print("[WARN] No results found. Run run_experiment.py first!")
    else:
        print(f"\nLoaded {len(all_results)} total results\n")

        # Parse all
        parsed = parse_all_results(all_results)

        # Show example
        if parsed and parsed[0]["parse_success"]:
            print("\nExample parsed result (first run):")
            print(f"Set ID: {parsed[0]['set_id']}")
            print(f"Mitigation: {parsed[0]['mitigation']}")
            print(f"Ranking: {parsed[0]['ranking']}")
            print(f"Scores: {parsed[0]['scores']}")
            print(f"Justifications keys: {list(parsed[0]['justifications'].keys())}")

        # Save parsed results
        output_path = Path(__file__).parent.parent / "results" / "parsed_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Saved parsed results to: {output_path}")
        print(f"[OK] Total parsed: {len(parsed)} ({sum(1 for p in parsed if p['parse_success'])} successful)")

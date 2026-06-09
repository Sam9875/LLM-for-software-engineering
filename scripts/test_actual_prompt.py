"""
Test with the actual prompt that failed in pilot_real.py.
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=".env.example", override=True)
key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("Base_url")

# Load the actual prompt
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import build_prompt
from generate_profiles import generate_profile_set

# Build the actual prompt like the pilot does
profile_set = generate_profile_set("test", 5)
prompt = build_prompt(profile_set, mitigation_strategy="baseline")

print(f"Prompt length: {len(prompt)} chars")
print(f"First 200 chars: {prompt[:200]}")
print("---")

# Send it the same way call_claude does
url = base_url + "/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2000,
    "temperature": 0.7,
}

print("Sending request with same params as call_claude()...")
try:
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    print(f"Status: {r.status_code}")
    print(f"Body (first 500 chars): {r.text[:500]}")
    print("---")
    if r.status_code == 200:
        data = r.json()
        if "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")
            print(f"SUCCESS. Response length: {len(content)}")
            print(f"First 300 chars: {content[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

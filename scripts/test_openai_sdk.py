"""
Test with the actual openai SDK (the way call_claude uses it).
Compare with raw requests to find the difference.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=".env.example", override=True)
key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("Base_url")

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import build_prompt
from generate_profiles import generate_profile_set

profile_set = generate_profile_set("test", 5)
prompt = build_prompt(profile_set, mitigation_strategy="baseline")

# Use openai SDK exactly like call_claude does
import openai
client = openai.OpenAI(api_key=key, base_url=base_url)

print("Testing with openai SDK...")
try:
    response = client.chat.completions.create(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
    )
    print(f"SUCCESS. Response: {response.choices[0].message.content[:200]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    # Print the request details if available
    if hasattr(e, 'request'):
        print(f"  request: {e.request}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"  response status: {e.response.status_code}")
        print(f"  response body: {e.response.text[:500]}")

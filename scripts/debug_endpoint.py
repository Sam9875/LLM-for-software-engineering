"""
Debug: figure out why the endpoint works for "pong" but not for the full prompt.
Tests different prompt sizes and shapes.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=".env.example", override=True)
key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("Base_url")

url = base_url + "/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

# Test 1: short prompt (like the pong test that worked)
test1 = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
}

# Test 2: medium prompt (about same size as our real prompt)
medium_text = "Please evaluate this candidate for an apartment. " * 50
test2 = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": medium_text}],
    "max_tokens": 2000,
}

# Test 3: long prompt (similar to our actual ~3700 char prompt)
long_text = "Please evaluate this candidate for an apartment. " * 100
test3 = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": long_text}],
    "max_tokens": 2000,
}

# Test 4: with the exact prompt size we use
test4 = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": "X" * 3680}],  # same size as our prompt
    "max_tokens": 2000,
}

for i, (name, payload) in enumerate([
    ("short prompt", test1),
    ("medium prompt (~1500 chars)", test2),
    ("long prompt (~3700 chars)", test3),
    ("exact size 3680 chars", test4),
], 1):
    print(f"\n[Test {i}] {name}")
    print(f"  prompt length: {len(payload['messages'][0]['content'])} chars")
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"  status: {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                if "choices" in data:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    print(f"  [OK] response: {content[:100]!r}")
                else:
                    print(f"  [WARN] no 'choices' in response: keys={list(data.keys())}")
            except Exception as e:
                print(f"  [WARN] json parse failed: {e}")
                print(f"  raw: {r.text[:200]}")
        else:
            print(f"  [X] FAILED")
            print(f"  body: {r.text[:300]}")
    except Exception as e:
        print(f"  [X] ERROR: {type(e).__name__}: {e}")

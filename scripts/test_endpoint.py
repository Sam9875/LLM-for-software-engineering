"""
Test if the custom API endpoint works.
Tries OpenAI-compatible protocol (since the URL was described that way).
Reports back success/failure without exposing any keys.
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ANTHROPIC_API_KEY")

# Fallback: try .env.example if .env doesn't have a key
if not key or key == "your_anthropic_api_key_here":
    load_dotenv(dotenv_path=".env.example", override=True)
    key = os.getenv("ANTHROPIC_API_KEY")

# Also read base_url from .env.example if set
base_url_env = None
load_dotenv(dotenv_path=".env.example", override=True)
base_url_env = os.getenv("Base_url") or os.getenv("BASE_URL") or os.getenv("base_url")

if not key:
    print("NO_KEY: ANTHROPIC_API_KEY not found in .env")
    sys.exit(1)

# Try OpenAI-compatible protocol
url = (base_url_env or "http://165.245.222.182:8000") + "/v1/chat/completions"

# Don't print the key, just confirm we have it
print(f"KEY_LOADED: yes (length {len(key)}, prefix {key[:8] if len(key) >= 8 else 'short'})")
print(f"ENDPOINT: {url}")
print(f"MODEL: claude-opus-4-8")
print("---")
headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "claude-opus-4-8",
    "messages": [{"role": "user", "content": "Reply with just the word: pong"}],
    "max_tokens": 10,
}

try:
    print("SENDING REQUEST...")
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    print(f"HTTP STATUS: {r.status_code}")
    print(f"RESPONSE BODY (first 500 chars):")
    print(r.text[:500])
    print("---")

    if r.status_code == 200:
        data = r.json()
        # Try to extract the response
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            print(f"SUCCESS: got response: {content!r}")
        else:
            print(f"UNEXPECTED SHAPE: keys = {list(data.keys())}")
    else:
        print(f"FAILED: status {r.status_code}")

except requests.exceptions.ConnectionError as e:
    print(f"CONNECTION_ERROR: cannot reach {url}")
    print(f"  Details: {type(e).__name__}")
    print("  -> This usually means: server is down, IP is wrong, or firewall is blocking")
except requests.exceptions.Timeout:
    print("TIMEOUT: server didn't respond in 30 seconds")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

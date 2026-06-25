#!/usr/bin/env python3
"""
Standalone chatbot script using Cerebras API.

The script sends a single user prompt to the Cerebras chat endpoint and
prints the assistant's response. No interactive input is required; the
prompt can be customized via the PROMPT variable or the
CEREBRAS_PROMPT environment variable.
"""

import os
import sys
import json
import requests

# Constants for the Cerebras API
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

def get_api_key():
    """Retrieve the Cerebras API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        print("Error: CEREBRAS_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return key

def build_payload(user_prompt: str) -> dict:
    """Construct the JSON payload for the chat request."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

def call_cerebras_api(api_key: str, payload: dict) -> dict:
    """Send the request to the Cerebras API and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"Error: API responded with status {response.status_code}", file=sys.stderr)
        try:
            err_detail = response.json()
            print("Response body:", json.dumps(err_detail, indent=2), file=sys.stderr)
        except Exception:
            print("Response body:", response.text, file=sys.stderr)
        sys.exit(1)
    return response.json()

def extract_reply(api_response: dict) -> str:
    """Extract the assistant's reply from the API response."""
    try:
        # Assuming OpenAI-compatible response format
        return api_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        print(f"Error: Unexpected response structure: {e}", file=sys.stderr)
        print("Full response:", json.dumps(api_response, indent=2), file=sys.stderr)
        sys.exit(1)

def main():
    # Determine the user prompt
    prompt = os.environ.get("CEREBRAS_PROMPT", "Hello! How are you today?")
    api_key = get_api_key()
    payload = build_payload(prompt)
    api_response = call_cerebras_api(api_key, payload)
    reply = extract_reply(api_response)
    print("Assistant:", reply)

if __name__ == "__main__":
    main()
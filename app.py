#!/usr/bin/env python3
import os
import sys
import json
import requests

def main():
    api_key = os.getenv('CEREBRAS_API_KEY')
    if not api_key:
        print("Error: CEREBRAS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    endpoint = "https://api.cerebras.ai/v1/chat/completions"
    model = "gpt-oss-120b"

    # Define a simple conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke about computers."}
    ]

    payload = {
        "model": model,
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        print(f"Error: Network request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print(f"Error: API request failed with status {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract the assistant's reply
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Error: Unexpected response structure: {e}", file=sys.stderr)
        sys.exit(1)

    print(reply.strip())

if __name__ == "__main__":
    main()
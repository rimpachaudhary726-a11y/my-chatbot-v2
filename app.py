#!/usr/bin/env python3
"""
A minimal chatbot client that talks to the Cerebras chat completion API.

The script expects a prompt either:
- as the first command‑line argument, e.g. `python chatbot.py "Hello!"`
- or via the environment variable `CHATBOT_PROMPT`
- if neither is provided, a default prompt is used so the script can run
  non‑interactively (e.g. in CI).

The Cerebras API key must be supplied via the environment variable
`CEREBRAS_API_KEY`. If the key is missing the script aborts with a clear
error message.

Typical usage (locally):
    export CEREBRAS_API_KEY=your_key_here
    python chatbot.py "What is the capital of France?"

Typical usage (GitHub Actions):
    - set `CEREBRAS_API_KEY` as a secret
    - optionally set `CHATBOT_PROMPT` or pass an argument to the step
"""

import os
import sys
import json
import textwrap

import requests  # external dependency

API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def get_prompt() -> str:
    """Return the prompt to send to the model.

    Priority:
    1. First CLI argument (everything after the script name)
    2. Environment variable CHATBOT_PROMPT
    3. Hard‑coded default (ensures the script runs without manual input)
    """
    if len(sys.argv) > 1:
        # Join all args so users can pass a multi‑word prompt without quoting
        return " ".join(sys.argv[1:])
    env_prompt = os.getenv("CHATBOT_PROMPT")
    if env_prompt:
        return env_prompt
    # Default prompt – a real request, not a fake placeholder
    return "Tell me a short joke about computers."


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.getenv("CEREBRAS_API_KEY")
    if not key:
        sys.stderr.write(
            "ERROR: CEREBRAS_API_KEY environment variable is not set.\n"
        )
        sys.exit(1)
    return key.strip()


def call_cerebras_api(prompt: str, api_key: str) -> str:
    """Send the prompt to Cerebras and return the assistant's reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        sys.stderr.write(f"ERROR: Network request failed: {exc}\n")
        sys.exit(1)

    if response.status_code != 200:
        sys.stderr.write(
            f"ERROR: API returned unexpected status {response.status_code}\n"
            f"Response body: {response.text}\n"
        )
        sys.exit(1)

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"ERROR: Failed to parse JSON response: {exc}\n")
        sys.exit(1)

    # The expected format follows OpenAI's chat completion schema:
    # {
    #   "choices": [{"message": {"role": "assistant", "content": "..."}}, ...],
    #   ...
    # }
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        sys.stderr.write(f"ERROR: Unexpected API response structure: {exc}\n")
        sys.stderr.write(f"Full response: {json.dumps(data, indent=2)}\n")
        sys.exit(1)

    return reply.strip()


def main() -> None:
    prompt = get_prompt()
    api_key = get_api_key()

    # Echo the prompt for visibility (useful in CI logs)
    print(f"Prompt: {prompt}\n", file=sys.stderr)

    reply = call_cerebras_api(prompt, api_key)

    # Pretty‑print the assistant's reply
    print("=== Assistant Reply ===")
    print(reply)


if __name__ == "__main__":
    main()
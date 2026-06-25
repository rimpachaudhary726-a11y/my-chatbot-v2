#!/usr/bin/env python3
"""
A minimal chatbot that forwards a user prompt to the Cerebras chat completion API
and prints the assistant's reply.

The script reads the required API key from the environment variable
CEREBRAS_API_KEY. If the variable is missing the script aborts with a clear error
message (exit code 1).

Usage (non‑interactive):
    python chatbot.py "Your question here"

No interactive input() is used, making it suitable for CI environments such as
GitHub Actions.
"""

import os
import sys
import json
from typing import Any, Dict, List

import requests

# ----------------------------------------------------------------------
# Configuration constants (do NOT modify unless you know the exact API spec)
# ----------------------------------------------------------------------
ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "gpt-oss-120b"


def fatal(msg: str) -> None:
    """Print an error message to stderr and exit with status 1."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        fatal("CEREBRAS_API_KEY environment variable is not set.")
    return key


def build_payload(user_prompt: str) -> Dict[str, Any]:
    """Create the JSON payload expected by the Cerebras chat completions endpoint."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    }


def call_cerebras_api(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST the payload to the Cerebras API and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        fatal(f"Network error while contacting Cerebras API: {exc}")

    if response.status_code != 200:
        # Attempt to include any JSON error returned by the API.
        try:
            err_content = response.json()
            err_msg = json.dumps(err_content, indent=2)
        except Exception:
            err_msg = response.text.strip()
        fatal(
            f"Cerebras API returned HTTP {response.status_code}:\n{err_msg}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        fatal(f"Failed to decode JSON response from Cerebras API: {exc}")


def extract_reply(api_response: Dict[str, Any]) -> str:
    """
    Pull the assistant's reply from the API response.
    Expected schema (compatible with OpenAI chat format):
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "..."},
                    ...
                },
                ...
            ],
            ...
        }
    """
    try:
        choices: List[Dict[str, Any]] = api_response["choices"]
        if not choices:
            fatal("API response contains no choices.")
        message = choices[0]["message"]
        content = message["content"]
        return content.strip()
    except (KeyError, TypeError) as exc:
        fatal(f"Unexpected API response structure: {exc}")


def main() -> None:
    # Ensure a prompt was supplied as a command‑line argument.
    if len(sys.argv) < 2:
        fatal(
            "No prompt supplied. Usage: python chatbot.py \"Your question here\""
        )
    # Join all arguments to allow spaces without quoting each word.
    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        fatal("Prompt is empty after stripping whitespace.")

    api_key = get_api_key()
    payload = build_payload(user_prompt)
    response_json = call_cerebras_api(api_key, payload)
    reply = extract_reply(response_json)

    print(reply)


if __name__ == "__main__":
    main()
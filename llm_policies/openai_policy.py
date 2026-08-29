#!/usr/bin/env python3
"""JSON-in/JSON-out OpenAI Responses API policy for research_agent.py."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.4-mini"


def build_payload(request: dict, model: str) -> dict:
    allowed = [item["experiment_id"] for item in request["allowed_experiments"]]
    if not allowed:
        raise ValueError("No allowed experiments were supplied")
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 512,
        "instructions": (
            "You are selecting the next safe recommender-system experiment. "
            "Use only supplied results and experiment descriptions. Prefer an "
            "experiment that is informative and likely to improve validation "
            "primary. Never request code, data, metric, or evaluator changes."
        ),
        "input": json.dumps(request, sort_keys=True),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "experiment_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "string", "enum": allowed},
                        "reason": {"type": "string"},
                    },
                    "required": ["experiment_id", "reason"],
                    "additionalProperties": False,
                },
            },
            "verbosity": "low",
        },
    }


def extract_output_text(response: dict) -> str:
    if response.get("output_text"):
        return response["output_text"]
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
    raise ValueError("Responses API returned no output text")


def read_policy_request(stream) -> dict:
    """Read JSON from Windows or Unix pipes, tolerating an optional UTF-8 BOM."""
    return json.loads(stream.buffer.read().decode("utf-8-sig"))


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    policy_request = read_policy_request(sys.stdin)
    payload = build_payload(policy_request, model)
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            api_response = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {detail}") from error

    choice = json.loads(extract_output_text(api_response))
    output = {
        "experiment_id": choice["experiment_id"],
        "reason": choice["reason"],
        "provider": "openai",
        "model": api_response.get("model", model),
        "usage": api_response.get("usage"),
        "response_id": api_response.get("id"),
    }
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

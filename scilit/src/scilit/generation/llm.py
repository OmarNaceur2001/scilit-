"""
SciLit - generation/llm.py
The one place that talks to the local LLM (Qwen2.5-7B-Instruct, 4-bit, served by Ollama).

  temperature 0      greedy decoding: same input -> same output (reproducible)
  format = schema    constrained decoding: at every step Ollama only allows tokens that keep the
                     output valid against the JSON schema, so the structure is guaranteed
  num_ctx 8192       context window; every prompt SciLit sends is built to fit in it
"""
from __future__ import annotations

import json

import requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


def ollama(prompt: str, schema: dict, model: str = MODEL) -> dict:
    r = requests.post(OLLAMA, json={"model": model, "stream": False, "format": schema,
                                    "options": {"temperature": 0, "num_ctx": 8192},
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])

"""
SciLit - generation/llm.py
The one place that talks to the local LLM (Qwen2.5-7B-Instruct, 4-bit, served by Ollama).

  temperature 0      greedy decoding: same input -> same output (reproducible)
  format = schema    constrained decoding: at every step Ollama only allows tokens that keep the
                     output valid against the JSON schema, so the structure is guaranteed
  num_ctx 8192       one context window for every call (changing it makes Ollama reload the model);
                     every prompt SciLit sends is built to fit in it
  model              SCILIT_LLM (default qwen2.5:7b; qwen3:8b for comparison, thinking disabled)
  parallel requests  the 8 papers are read by concurrent requests; Ollama runs them at the same time
                     when OLLAMA_NUM_PARALLEL > 1 (and GPU memory allows), otherwise it queues them
"""
from __future__ import annotations

import json
import os

import requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = os.environ.get("SCILIT_LLM", "qwen2.5:7b")             # or "qwen3:8b"


def ollama(prompt: str, schema: dict, model: str = MODEL) -> dict:
    """Structured call: the answer is JSON valid against `schema`."""
    body = {"model": model, "stream": False, "format": schema, "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt}]}
    if model.startswith("qwen3"):
        body["think"] = False                    # Qwen3: answer directly, no hidden reasoning trace
    r = requests.post(OLLAMA, json=body, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def ollama_text(prompt: str, model: str) -> str:
    """Plain completion (used by the MiniCheck verifier, which answers "Yes" or "No")."""
    r = requests.post(OLLAMA.replace("/api/chat", "/api/generate"), json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 3}}, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

"""Download every Hugging Face model SciLit needs, once (cached in ~/.cache/huggingface).
Only one weight format per model (safetensors if available) to avoid downloading twice."""
from huggingface_hub import list_repo_files, snapshot_download

MODELS = [
    "BAAI/bge-reranker-v2-m3",                        # Part 6: cross-encoder reranking
    "BAAI/bge-small-en-v1.5",                         # Part 6: baseline embedding model
    "allenai/scibert_scivocab_uncased",               # Parts 7-8: scientific encoder
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",   # Part 11: NLI verification
]
for m in MODELS:
    files = list_repo_files(m)
    weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
    snapshot_download(m, allow_patterns=["*.json", "*.txt", "*.model", weights])
    print("ok", m, flush=True)

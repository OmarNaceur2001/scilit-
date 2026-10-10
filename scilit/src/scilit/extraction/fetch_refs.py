"""
SciLit — extraction/fetch_refs.py
====================================
Étape 3 : Récupère referenced_works depuis OpenAlex pour tous les articles.

Bronze : interroge OpenAlex via source.arxiv_id (50 IDs par batch).
         Les articles sans arxiv_id sont enregistrés dans refs_skipped.jsonl.

Sortie :
  data/parsed/refs_raw.jsonl   — une ligne par article trouvé
    {"scilit_uuid": "...", "arxiv_id": "...", "openalex_id": "W...", "references": ["W...", ...]}
  data/parsed/refs_skipped.jsonl — articles non interrogeables (raison explicitée)

Usage :
  python -m scilit.extraction.fetch_refs
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

RAW_ARTICLES  = Path("data/raw/articles.jsonl")
OUT_REFS      = Path("data/parsed/refs_raw.jsonl")
OUT_SKIPPED   = Path("data/parsed/refs_skipped.jsonl")
OA_API        = "https://api.openalex.org/works"
BATCH_SIZE    = 50
SLEEP_BETWEEN = 0.2
MAX_RETRIES   = 3


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_articles() -> list[dict]:
    return [
        json.loads(l)
        for l in RAW_ARTICLES.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _done_uuids() -> set[str]:
    """UUIDs already in refs_raw OR refs_skipped — pour reprendre sans doublons."""
    done: set[str] = set()
    for path in (OUT_REFS, OUT_SKIPPED):
        if path.exists():
            for l in path.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        done.add(json.loads(l)["scilit_uuid"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def _query_batch(arxiv_ids: list[str]) -> list[dict]:
    """
    Une requête OpenAlex pour jusqu'à 50 arxiv IDs via leur DOI arXiv.
    Retourne la liste brute de works de l'API.
    """
    params = {
        "filter":   "doi:" + "|".join(f"10.48550/arxiv.{aid}" for aid in arxiv_ids),
        "select":   "id,locations,referenced_works",
        "per-page": str(len(arxiv_ids)),
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(OA_API, params=params, timeout=30)
            if r.status_code == 429:
                wait = 5 * attempt
                log.warning("429 — attente %ds (tentative %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.RequestException as e:
            log.warning("Tentative %d/%d échouée : %s", attempt, MAX_RETRIES, e)
            time.sleep(2 * attempt)
    log.error("Batch échoué après %d tentatives.", MAX_RETRIES)
    return []


def _parse_arxiv_id(work: dict) -> str | None:
    """
    Extrait l'arxiv ID depuis les locations d'un work OpenAlex.
    """
    for loc in work.get("locations", []):
        url = loc.get("landing_page_url", "") or ""
        if "arxiv.org/abs/" in url:
            return url.rstrip("/").split("/")[-1]
        if "10.48550/arxiv." in url.lower():
            return url.rstrip("/").split("10.48550/arxiv.")[-1]
    return None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    articles = _load_articles()
    done     = _done_uuids()
    log.info("%d articles chargés, %d déjà traités.", len(articles), len(done))

    # ── Séparer avec / sans arxiv_id ───────────────────────────────────────
    with_arxiv:    list[dict] = []
    without_arxiv: list[dict] = []
    for a in articles:
        if a["id"] in done:
            continue
        if a.get("source", {}).get("arxiv_id"):
            with_arxiv.append(a)
        else:
            without_arxiv.append(a)

    log.info(
        "%d à traiter (arxiv_id présent), %d ignorés (pas d'arxiv_id).",
        len(with_arxiv), len(without_arxiv),
    )

    # ── Enregistrer les articles sans arxiv_id ─────────────────────────────
    with OUT_SKIPPED.open("a", encoding="utf-8") as sf:
        for a in without_arxiv:
            sf.write(json.dumps({
                "scilit_uuid": a["id"],
                "title":       a["title"][:80],
                "reason":      "no_arxiv_id",
            }) + "\n")

    # ── Index arxiv_id → article ───────────────────────────────────────────
    arxiv_map: dict[str, dict] = {a["source"]["arxiv_id"]: a for a in with_arxiv}
    arxiv_ids = list(arxiv_map.keys())

    found     = 0
    not_found = 0

    # ── Requêtes par batch ─────────────────────────────────────────────────
    with OUT_REFS.open("a", encoding="utf-8") as rf, \
         OUT_SKIPPED.open("a", encoding="utf-8") as sf:

        for i in tqdm(range(0, len(arxiv_ids), BATCH_SIZE),
                      desc="OpenAlex batches", unit="batch", ncols=80):
            batch   = arxiv_ids[i : i + BATCH_SIZE]
            results = _query_batch(batch)

            returned: set[str] = set()
            for w in results:
                arxiv_id = _parse_arxiv_id(w)
                if not arxiv_id or arxiv_id not in arxiv_map:
                    continue
                article  = arxiv_map[arxiv_id]
                oa_id    = w.get("id", "").split("/")[-1]   # "W2741809807"
                refs     = [r.split("/")[-1] for r in w.get("referenced_works", [])]
                rf.write(json.dumps({
                    "scilit_uuid":  article["id"],
                    "arxiv_id":     arxiv_id,
                    "openalex_id":  oa_id,
                    "references":   refs,
                }) + "\n")
                returned.add(arxiv_id)
                found += 1

            # Articles du batch non retournés par OpenAlex
            for aid in set(batch) - returned:
                sf.write(json.dumps({
                    "scilit_uuid": arxiv_map[aid]["id"],
                    "title":       arxiv_map[aid]["title"][:80],
                    "reason":      "not_found_on_openalex",
                }) + "\n")
                not_found += 1

            time.sleep(SLEEP_BETWEEN)

    print(f"\n{'='*55}")
    print(f"  refs_raw.jsonl     : {found} articles avec références")
    print(f"  Articles ignorés   : {len(without_arxiv) + not_found}")
    print(f"    dont sans arxiv  : {len(without_arxiv)}")
    print(f"    dont non trouvés : {not_found}")
    print(f"{'='*55}")
    print("\nProchain : python -m scilit.extraction.citations")


if __name__ == "__main__":
    main()
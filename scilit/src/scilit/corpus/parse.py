"""
SciLit -- corpus/parse.py
=========================
Transforme les Article en Section + Chunk.

Bronze (--mode abstract) : abstract seul -> 1 Section + N Chunks
  - Pas de PDF, pas de GROBID
  - Suffisant pour Bronze W7 (BM25, baselines, demo)
  - Tourne en < 1 min sur 530 articles

Silver (--mode pdf) : telechargement PDF + PyMuPDF -> sections detectees
  - Telecharge les PDFs arXiv (accès ouvert)
  - Extraction texte + detection de sections par heuristiques
  - SciBERT viendra plus tard remplacer les heuristiques (Lot C)

Sortie :
  data/parsed/sections.jsonl   -- un Section JSON par ligne
  data/parsed/chunks.jsonl     -- un Chunk JSON par ligne
  data/parsed/parse_log.txt    -- journal

Usage :
  python -m scilit.corpus.parse                     # Bronze (abstract)
  python -m scilit.corpus.parse --mode pdf          # Silver (PDF)
  python -m scilit.corpus.parse --mode pdf --max 50 # Silver test
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from uuid import UUID

import requests
from tqdm import tqdm

from scilit.corpus.collect import ARTICLES_PATH, load_articles
from scilit.schemas import Article, Chunk, Section, SectionType

# =============================================================================
# CONFIG
# =============================================================================

PARSED_DIR     = Path("data/parsed")
SECTIONS_PATH  = PARSED_DIR / "sections.jsonl"
CHUNKS_PATH    = PARSED_DIR / "chunks.jsonl"
LOG_PATH       = PARSED_DIR / "parse_log.txt"
PDF_CACHE_DIR  = Path("data/raw/pdfs")

# Chunking
CHUNK_SIZE_CHARS    = 1000   # ~250 tokens (estimation : 4 chars/token)
CHUNK_OVERLAP_CHARS = 200    # ~50 tokens d'overlap

# Silver : telechargement PDF
PDF_DELAY      = 2.0         # secondes entre telechargements
PDF_TIMEOUT    = 30
MAX_PDF_SIZE   = 20_000_000  # 20 MB max

# Heuristiques de detection de sections (Silver)
SECTION_PATTERNS: list[tuple[re.Pattern, SectionType]] = [
    (re.compile(r"^abstract\b",              re.I), SectionType.ABSTRACT),
    (re.compile(r"^1\.?\s*introduction\b",   re.I), SectionType.INTRODUCTION),
    (re.compile(r"^related\s+work\b",        re.I), SectionType.RELATED_WORK),
    (re.compile(r"^background\b",            re.I), SectionType.RELATED_WORK),
    (re.compile(r"^(2|3)\.?\s*(method|approach|model|system)\b", re.I), SectionType.METHOD),
    (re.compile(r"^(3|4)\.?\s*(experiment|setup|dataset)\b",     re.I), SectionType.EXPERIMENT),
    (re.compile(r"^(4|5)\.?\s*(result|evaluation|analysis)\b",   re.I), SectionType.RESULTS),
    (re.compile(r"^discussion\b",            re.I), SectionType.DISCUSSION),
    (re.compile(r"^(5|6|7)\.?\s*conclusion", re.I), SectionType.CONCLUSION),
    (re.compile(r"^limitation\b",            re.I), SectionType.LIMITATIONS),
    (re.compile(r"^acknowledgement\b",       re.I), SectionType.ACKNOWLEDGEMENTS),
    (re.compile(r"^reference\b",             re.I), SectionType.REFERENCES),
    (re.compile(r"^appendix\b",              re.I), SectionType.APPENDIX),
]

# =============================================================================
# LOGGING
# =============================================================================

PARSED_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(
    text: str,
    article_id: UUID,
    section_id: UUID,
    section_type: SectionType,
    size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Decoupe un texte en Chunks avec overlap.
    Respecte les frontieres de phrases quand possible.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    idx   = 0
    n     = len(text)

    while start < n:
        end = min(start + size, n)

        # Chercher la fin de phrase la plus proche avant 'end'
        if end < n:
            # Cherche un '.' ou '\n' dans les 100 derniers chars
            boundary = text.rfind(".", start + size - 100, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + size - 100, end)
            if boundary != -1:
                end = boundary + 1

        fragment = text[start:end].strip()

        if len(fragment) >= 30:   # ignorer les fragments trop courts
            chunks.append(Chunk(
                article_id=article_id,
                section_id=section_id,
                section_type=section_type,
                text=fragment,
                char_start=start,
                char_end=end,
                chunk_index=idx,
                token_count=len(fragment) // 4,   # estimation
            ))
            idx += 1

        # Avancer avec overlap
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(size - overlap, 1)
        start = next_start

    return chunks

# =============================================================================
# BRONZE : ABSTRACT SEULEMENT
# =============================================================================

def parse_abstract(article: Article) -> tuple[list[Section], list[Chunk]]:
    """
    Bronze : cree 1 Section ABSTRACT + N Chunks depuis article.abstract.
    Aucun telechargement, aucune dependance externe.
    """
    abstract = article.abstract.strip()
    if not abstract:
        return [], []

    section = Section(
        article_id=article.id,
        section_type=SectionType.ABSTRACT,
        title="Abstract",
        text=abstract,
        char_start=0,
        char_end=len(abstract),
        section_index=0,
    )
    chunks = chunk_text(
        text=abstract,
        article_id=article.id,
        section_id=section.id,
        section_type=SectionType.ABSTRACT,
    )
    return [section], chunks


def run_bronze(
    articles: list[Article],
    sections_path: Path = SECTIONS_PATH,
    chunks_path:   Path = CHUNKS_PATH,
) -> tuple[int, int]:
    """
    Bronze : abstract -> Section + Chunk pour tous les articles.
    Append-safe : relancer sans dupliquer.
    """
    # Charger les article_ids deja traites
    done_ids: set[str] = set()
    if sections_path.exists():
        with open(sections_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        import json
                        d = json.loads(line)
                        done_ids.add(str(d.get("article_id", "")))
                    except Exception:
                        pass
    if done_ids:
        log.info(f"Bronze : {len(done_ids)} articles deja traites, reprise...")

    total_s = total_c = 0
    mode_s  = "a" if sections_path.exists() else "w"
    mode_c  = "a" if chunks_path.exists()   else "w"

    with (
        open(sections_path, mode_s, encoding="utf-8") as fs,
        open(chunks_path,   mode_c, encoding="utf-8") as fc,
    ):
        for article in tqdm(articles, desc="Bronze parse", unit="art", ncols=80):
            if str(article.id) in done_ids:
                continue

            sections, chunks = parse_abstract(article)

            for s in sections:
                fs.write(s.model_dump_json() + "\n")
                total_s += 1

            for c in chunks:
                fc.write(c.model_dump_json(exclude={"embedding"}) + "\n")
                total_c += 1

    return total_s, total_c

# =============================================================================
# SILVER : PDF + PyMuPDF
# =============================================================================

def _pdf_url_for(article: Article) -> str | None:
    """Retourne l'URL PDF open-access si disponible."""
    if article.source.pdf_url:
        return str(article.source.pdf_url)
    if article.source.arxiv_id:
        return f"https://arxiv.org/pdf/{article.source.arxiv_id}"
    return None


def _download_pdf(url: str, dest: Path) -> bool:
    """Telecharge un PDF dans dest. Retourne True si reussi."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, timeout=PDF_TIMEOUT, stream=True)
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                size += len(chunk)
                if size > MAX_PDF_SIZE:
                    log.warning(f"PDF trop grand (> 20 MB), abandon : {url}")
                    dest.unlink(missing_ok=True)
                    return False
        return True
    except Exception as e:
        log.debug(f"Telechargement PDF echoue ({url}): {e}")
        dest.unlink(missing_ok=True)
        return False


def _detect_section_type(title_line: str) -> SectionType:
    """Detecte le type de section depuis un titre."""
    t = title_line.strip()
    for pattern, stype in SECTION_PATTERNS:
        if pattern.match(t):
            return stype
    return SectionType.OTHER


def _extract_sections_pymupdf(pdf_path: Path, article: Article) -> list[Section]:
    """
    Extrait le texte du PDF avec PyMuPDF et segmente en sections par
    heuristiques (taille de police et patterns de titres).
    SciBERT remplacera ceci au palier Silver-avance (Lot C).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error("PyMuPDF non installe : pip install pymupdf")
        return []

    sections: list[Section] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.warning(f"Impossible d'ouvrir le PDF ({pdf_path.name}): {e}")
        return []

    # ── Extraire les blocs avec leur taille de police ──────────────────────
    blocks_by_font: list[tuple[float, str]] = []  # (font_size, text)
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
        for block in blocks:
            if block.get("type") != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text:
                        blocks_by_font.append((size, text))

    doc.close()

    if not blocks_by_font:
        return []

    # ── Trouver la taille de police des titres (heuristique) ───────────────
    sizes = [s for s, _ in blocks_by_font if s > 0]
    if not sizes:
        return []
    body_size  = sorted(sizes)[len(sizes) // 2]   # mediane = corps du texte
    title_size = body_size * 1.15                  # titres = 15% plus grands

    # ── Segmenter en sections ──────────────────────────────────────────────
    current_type  = SectionType.OTHER
    current_title = None
    current_text: list[str] = []
    current_start = 0
    char_pos = 0
    section_idx = 0

    def _flush(end_pos: int) -> None:
        nonlocal section_idx
        text = " ".join(current_text).strip()
        if len(text) < 30:
            return
        sections.append(Section(
            article_id=article.id,
            section_type=current_type,
            title=current_title,
            text=text,
            char_start=current_start,
            char_end=end_pos,
            section_index=section_idx,
        ))
        section_idx += 1

    for font_size, text in blocks_by_font:
        is_title = font_size >= title_size and len(text) < 120

        if is_title:
            _flush(char_pos)
            current_type  = _detect_section_type(text)
            current_title = text
            current_text  = []
            current_start = char_pos
        else:
            current_text.append(text)

        char_pos += len(text) + 1

    _flush(char_pos)

    # Fallback : si aucune section detectee, abstract uniquement
    if not sections:
        abstract = article.abstract.strip()
        if abstract:
            sections.append(Section(
                article_id=article.id,
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                text=abstract,
                char_start=0,
                char_end=len(abstract),
                section_index=0,
            ))

    return sections


def run_silver(
    articles: list[Article],
    max_articles: int = 100,
    sections_path: Path = SECTIONS_PATH,
    chunks_path:   Path = CHUNKS_PATH,
) -> tuple[int, int]:
    """
    Silver : telecharge les PDFs et extrait les sections avec PyMuPDF.
    Lance seulement sur max_articles pour eviter de surcharger le disque.
    """
    log.info(f"Silver : traitement de {min(max_articles, len(articles))} articles...")
    total_s = total_c = downloaded = failed = 0

    mode_s = "a" if sections_path.exists() else "w"
    mode_c = "a" if chunks_path.exists()   else "w"

    with (
        open(sections_path, mode_s, encoding="utf-8") as fs,
        open(chunks_path,   mode_c, encoding="utf-8") as fc,
    ):
        for article in tqdm(articles[:max_articles], desc="Silver PDF", ncols=80):
            pdf_url = _pdf_url_for(article)
            if not pdf_url:
                # Fallback Bronze
                sections, chunks = parse_abstract(article)
            else:
                safe_name = article.corpus_id + ".pdf"
                pdf_path  = PDF_CACHE_DIR / safe_name

                ok = _download_pdf(pdf_url, pdf_path)
                if ok:
                    downloaded += 1
                    sections = _extract_sections_pymupdf(pdf_path, article)
                    if not sections:
                        sections, _ = parse_abstract(article)
                else:
                    failed += 1
                    sections, _ = parse_abstract(article)

                time.sleep(PDF_DELAY)

            # Chunks depuis chaque section
            chunks: list[Chunk] = []
            for s in sections:
                chunks.extend(chunk_text(
                    text=s.text,
                    article_id=article.id,
                    section_id=s.id,
                    section_type=s.section_type,
                ))

            for s in sections:
                fs.write(s.model_dump_json() + "\n")
                total_s += 1
            for c in chunks:
                fc.write(c.model_dump_json(exclude={"embedding"}) + "\n")
                total_c += 1

    log.info(f"Silver : PDF ok={downloaded} echec={failed}")
    return total_s, total_c

# =============================================================================
# UTILITAIRES DE LECTURE
# =============================================================================

def load_sections(path: Path = SECTIONS_PATH) -> list[Section]:
    """Charge toutes les Sections depuis sections.jsonl."""
    items = []
    if not path.exists():
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(Section.model_validate_json(line))
            except Exception:
                pass
    log.info(f"{len(items)} sections chargees.")
    return items


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    """Charge tous les Chunks depuis chunks.jsonl."""
    items = []
    if not path.exists():
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(Chunk.model_validate_json(line))
            except Exception:
                pass
    log.info(f"{len(items)} chunks charges.")
    return items


def print_stats(sections_path: Path, chunks_path: Path) -> None:
    """Affiche les statistiques du parsing."""
    sections = load_sections(sections_path)
    chunks   = load_chunks(chunks_path)

    sec_types: dict[str, int] = {}
    for s in sections:
        k = s.section_type.value
        sec_types[k] = sec_types.get(k, 0) + 1

    token_total = sum(c.token_count or 0 for c in chunks)
    articles    = len({str(s.article_id) for s in sections})

    print(f"\n{'='*55}")
    print(f"  PARSING : {articles} articles traites")
    print(f"  Sections : {len(sections)} | Chunks : {len(chunks)}")
    print(f"  Tokens estimes : {token_total:,}")
    print(f"  Types sections : {sec_types}")
    avg = len(chunks) // articles if articles else 0
    print(f"  Chunks/article : ~{avg}")
    print(f"{'='*55}\n")

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="SciLit -- parsing articles")
    p.add_argument(
        "--mode",
        choices=["abstract", "pdf"],
        default="abstract",
        help="abstract = Bronze (rapide) | pdf = Silver (PyMuPDF)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        help="Nb max d'articles a traiter (default: tous)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ARTICLES_PATH,
        help=f"Chemin vers articles.jsonl (default: {ARTICLES_PATH})",
    )
    args = p.parse_args()

    articles = load_articles(args.input)
    if not articles:
        print("Aucun article trouve. Lance d'abord : python -m scilit.corpus.collect")
        return

    if args.max:
        articles = articles[:args.max]

    log.info(f"Mode : {args.mode} | Articles : {len(articles)}")

    if args.mode == "abstract":
        ns, nc = run_bronze(articles)
    else:
        ns, nc = run_silver(articles, max_articles=args.max or len(articles))

    log.info(f"Sections sauvegardees : {ns} | Chunks : {nc}")
    print_stats(SECTIONS_PATH, CHUNKS_PATH)

if __name__ == "__main__":
    main()
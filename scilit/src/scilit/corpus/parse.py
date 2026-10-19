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
from uuid import UUID, uuid5

import requests
from tqdm import tqdm

from scilit import config
from scilit.corpus.collect import load_articles
from scilit.schemas import Article, Chunk, Section, SectionType

# =============================================================================
# CONFIG
# =============================================================================

PARSED_DIR     = config.PARSED_DIR
SECTIONS_PATH  = PARSED_DIR / "sections.jsonl"
CHUNKS_PATH    = PARSED_DIR / "chunks.jsonl"
LOG_PATH       = PARSED_DIR / "parse_log.txt"
PDF_CACHE_DIR  = config.RAW_DIR / "pdfs"

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

log = logging.getLogger(__name__)


def resolve_full_text_mode(
    mode: str | None,
    full_text_mode: bool | None,
) -> bool:
    """Resolve the legacy mode and explicit full-text flag consistently."""
    if mode not in (None, "abstract", "pdf"):
        raise ValueError("mode must be 'abstract' or 'pdf'")

    legacy_full_text = None if mode is None else mode == "pdf"
    if (
        legacy_full_text is not None
        and full_text_mode is not None
        and legacy_full_text != full_text_mode
    ):
        raise ValueError(
            "--mode and --full-text-mode specify conflicting parsing modes"
        )

    if full_text_mode is not None:
        return full_text_mode
    return legacy_full_text if legacy_full_text is not None else False


def build_argument_parser(input_default: Path | None = None) -> argparse.ArgumentParser:
    """Build the parser CLI while retaining the original --mode interface."""
    parser = argparse.ArgumentParser(description="SciLit -- parsing articles")
    parser.add_argument(
        "--mode",
        choices=["abstract", "pdf"],
        default=None,
        help="Legacy mode selector: abstract (default) or pdf (full text)",
    )
    parser.add_argument(
        "--full-text-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly select full-text PDF parsing (use --no-full-text-mode for abstracts)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Nb max d'articles a traiter (default: tous)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=input_default,
        help=(
            f"Chemin vers articles.jsonl (default: {input_default})"
            if input_default is not None
            else "Chemin vers articles.jsonl (default: configuration SciLit)"
        ),
    )
    return parser

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

        # Prefer a sentence boundary. If none is nearby, stop at a word boundary
        # so a long sentence cannot be cut in the middle of a token.
        if end < n:
            lower_bound = max(start + 1, end - 100)
            boundary = max(
                text.rfind(mark, lower_bound, end)
                for mark in (".", "!", "?", "\n")
            )
            if boundary != -1:
                end = boundary + 1
            else:
                word_boundary = text.rfind(" ", lower_bound, end)
                if word_boundary > start:
                    end = word_boundary

        raw_fragment = text[start:end]
        left_trim = len(raw_fragment) - len(raw_fragment.lstrip())
        right_trimmed = raw_fragment.rstrip()
        fragment = right_trimmed[left_trim:]
        fragment_start = start + left_trim
        fragment_end = start + len(right_trimmed)

        if len(fragment) >= 30:   # ignorer les fragments trop courts
            chunks.append(Chunk(
                id=uuid5(section_id, f"chunk:{idx}"),
                article_id=article_id,
                section_id=section_id,
                section_type=section_type,
                text=fragment,
                char_start=fragment_start,
                char_end=fragment_end,
                chunk_index=idx,
                token_count=len(fragment) // 4,   # estimation
            ))
            idx += 1

        # Avancer avec overlap on a word boundary. The first sentence of an
        # overlapping chunk is discarded by claim extraction when it starts
        # inside a section; the preceding chunk owns that passage.
        next_start = end - overlap
        while (
            next_start < end
            and next_start > start
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            next_start += 1
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
        id=uuid5(article.id, "section:0"),
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
    sections_path: Path | None = None,
    chunks_path: Path | None = None,
) -> tuple[int, int]:
    """
    Bronze : abstract -> Section + Chunk pour tous les articles.
    Append-safe : relancer sans dupliquer.
    """
    paths = config.paths()
    sections_path = sections_path or paths.parsed / "sections.jsonl"
    chunks_path = chunks_path or paths.parsed / "chunks.jsonl"
    sections_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

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
    processed = failed = skipped = 0
    mode_s  = "a" if sections_path.exists() else "w"
    mode_c  = "a" if chunks_path.exists()   else "w"

    with (
        open(sections_path, mode_s, encoding="utf-8") as fs,
        open(chunks_path,   mode_c, encoding="utf-8") as fc,
    ):
        for article in tqdm(articles, desc="Bronze parse", unit="art", ncols=80):
            if str(article.id) in done_ids:
                skipped += 1
                continue

            sections, chunks = parse_abstract(article)
            if sections:
                processed += 1
            else:
                failed += 1

            for s in sections:
                fs.write(s.model_dump_json() + "\n")
                total_s += 1

            for c in chunks:
                fc.write(c.model_dump_json(exclude={"embedding"}) + "\n")
                total_c += 1

    log.info(
        "Parse summary: requested_mode=abstract effective_mode=abstract "
        "input_articles=%d processed_abstract=%d processed_full_text=0 "
        "failed=%d skipped_existing=%d skipped_no_pdf=0 "
        "sections=%d chunks=%d",
        len(articles),
        processed,
        failed,
        skipped,
        total_s,
        total_c,
    )
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
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            cached_size = dest.stat().st_size
            if 1000 < cached_size <= MAX_PDF_SIZE:
                return True
            if cached_size > MAX_PDF_SIZE:
                log.warning("Cached PDF exceeds size limit and was discarded: %s", dest)
                dest.unlink(missing_ok=True)
                return False
        dest.unlink(missing_ok=True)
        r = requests.get(url, timeout=PDF_TIMEOUT, stream=True)
        r.raise_for_status()
        size = 0
        too_large = False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if size + len(chunk) > MAX_PDF_SIZE:
                    too_large = True
                    break
                f.write(chunk)
                size += len(chunk)
        if too_large:
            log.warning("PDF exceeds size limit; download discarded: %s", url)
            dest.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:
        log.warning("PDF download failed (%s): %s", url, e)
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
    try:
        for page in doc:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)[
                "blocks"
            ]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        if text:
                            blocks_by_font.append((size, text))
    finally:
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
            id=uuid5(article.id, f"section:{section_idx}"),
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

    return sections


def run_silver(
    articles: list[Article],
    max_articles: int = 100,
    sections_path: Path | None = None,
    chunks_path: Path | None = None,
) -> tuple[int, int]:
    """
    Silver : telecharge les PDFs et extrait les sections avec PyMuPDF.
    Lance seulement sur max_articles pour eviter de surcharger le disque.
    """
    paths = config.paths()
    sections_path = sections_path or paths.parsed / "sections.jsonl"
    chunks_path = chunks_path or paths.parsed / "chunks.jsonl"
    sections_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    selected = articles[:max_articles]
    log.info(
        "Starting full-text parsing: input_articles=%d selected_articles=%d",
        len(articles),
        len(selected),
    )
    total_s = total_c = pdf_available = processed = failed = skipped_no_pdf = 0

    mode_s = "a" if sections_path.exists() else "w"
    mode_c = "a" if chunks_path.exists() else "w"

    with (
        open(sections_path, mode_s, encoding="utf-8") as fs,
        open(chunks_path, mode_c, encoding="utf-8") as fc,
    ):
        for article in tqdm(selected, desc="Full-text PDF", ncols=80):
            pdf_url = _pdf_url_for(article)
            if not pdf_url:
                skipped_no_pdf += 1
                log.warning(
                    "Full-text skipped: article_id=%s reason=no_pdf_url",
                    article.id,
                )
                continue

            pdf_path = paths.raw / "pdfs" / f"{article.corpus_id}.pdf"
            try:
                if not _download_pdf(pdf_url, pdf_path):
                    failed += 1
                    log.error(
                        "Full-text failed: article_id=%s reason=pdf_download_failed",
                        article.id,
                    )
                    continue

                pdf_available += 1
                sections = _extract_sections_pymupdf(pdf_path, article)
                if not sections:
                    failed += 1
                    pdf_path.unlink(missing_ok=True)
                    log.error(
                        "Full-text failed: article_id=%s reason=no_extractable_sections",
                        article.id,
                    )
                    continue

                processed += 1
                log.info(
                    "Full-text processed: article_id=%s sections=%d",
                    article.id,
                    len(sections),
                )
            except Exception:
                failed += 1
                pdf_path.unlink(missing_ok=True)
                log.exception(
                    "Full-text failed: article_id=%s reason=extraction_error",
                    article.id,
                )
                continue
            finally:
                time.sleep(PDF_DELAY)

            chunks: list[Chunk] = []
            for section in sections:
                chunks.extend(
                    chunk_text(
                        text=section.text,
                        article_id=article.id,
                        section_id=section.id,
                        section_type=section.section_type,
                    )
                )

            for section in sections:
                fs.write(section.model_dump_json() + "\n")
                total_s += 1
            for chunk in chunks:
                fc.write(chunk.model_dump_json(exclude={"embedding"}) + "\n")
                total_c += 1

    log.info(
        "Parse summary: requested_mode=full_text effective_mode=full_text "
        "input_articles=%d selected_articles=%d processed_abstract=0 "
        "processed_full_text=%d failed=%d skipped_no_pdf=%d "
        "pdf_available=%d sections=%d chunks=%d",
        len(articles),
        len(selected),
        processed,
        failed,
        skipped_no_pdf,
        pdf_available,
        total_s,
        total_c,
    )
    return total_s, total_c

# =============================================================================
# UTILITAIRES DE LECTURE
# =============================================================================

def load_sections(path: Path | None = None) -> list[Section]:
    """Charge toutes les Sections depuis sections.jsonl."""
    path = path or config.paths().parsed / "sections.jsonl"
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


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Charge tous les Chunks depuis chunks.jsonl."""
    path = path or config.paths().parsed / "chunks.jsonl"
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
    print("  EXISTING AGGREGATE OUTPUT (may contain prior runs/modes)")
    print(f"  Articles with sections : {articles}")
    print(f"  Sections : {len(sections)} | Chunks : {len(chunks)}")
    print(f"  Tokens estimes : {token_total:,}")
    print(f"  Types sections : {sec_types}")
    avg = len(chunks) // articles if articles else 0
    print(f"  Chunks/article : ~{avg}")
    print(f"{'='*55}\n")

# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    paths = config.paths()
    config.setup_logging(paths.parsed / "parse_log.txt")
    parser = build_argument_parser(paths.raw / "articles.jsonl")
    args = parser.parse_args(argv)
    try:
        full_text_mode = resolve_full_text_mode(args.mode, args.full_text_mode)
    except ValueError as exc:
        parser.error(str(exc))

    articles = load_articles(args.input)
    if not articles:
        print("Aucun article trouve. Lance d'abord : python -m scilit.corpus.collect")
        return

    input_count = len(articles)
    selected_articles = articles if args.max is None else articles[:args.max]
    requested_mode = args.mode or (
        "full_text"
        if args.full_text_mode is True
        else "abstract"
        if args.full_text_mode is False
        else "default"
    )
    effective_mode = "full_text" if full_text_mode else "abstract"
    log.info(
        "Parse request: requested_mode=%s full_text_mode_flag=%s "
        "effective_mode=%s input_articles=%d selected_articles=%d",
        requested_mode,
        args.full_text_mode if args.full_text_mode is not None else "default",
        effective_mode,
        input_count,
        len(selected_articles),
    )

    if not full_text_mode:
        ns, nc = run_bronze(selected_articles)
    else:
        max_articles = args.max if args.max is not None else input_count
        ns, nc = run_silver(articles, max_articles=max_articles)

    log.info(f"Sections sauvegardees : {ns} | Chunks : {nc}")
    print_stats(paths.parsed / "sections.jsonl", paths.parsed / "chunks.jsonl")

if __name__ == "__main__":
    main()

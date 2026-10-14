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
    (re.compile(r"^intro(?:duction)?\b",     re.I), SectionType.INTRODUCTION),
    (re.compile(r"^related\s+work\b",        re.I), SectionType.RELATED_WORK),
    (re.compile(r"^background\b",            re.I), SectionType.RELATED_WORK),
    (re.compile(r"^(methods?|approach|model|system)\b", re.I), SectionType.METHOD),
    (re.compile(r"^(experiments?|setup|datasets?|data)\b", re.I), SectionType.EXPERIMENT),
    (re.compile(r"^(results?|evaluation|analysis)\b", re.I), SectionType.RESULTS),
    (re.compile(r"^discussions?\b", re.I), SectionType.DISCUSSION),
    (re.compile(r"^conclusion\b",             re.I), SectionType.CONCLUSION),
    (re.compile(r"^limitations?\b",          re.I), SectionType.LIMITATIONS),
    (re.compile(r"^acknowledgements?\b", re.I), SectionType.ACKNOWLEDGEMENTS),
    (re.compile(r"^references?\b|^bibliography\b", re.I), SectionType.REFERENCES),
    (re.compile(r"^append(?:ix|ices)\b", re.I), SectionType.APPENDIX),
]

# =============================================================================
# LOGGING
# =============================================================================

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
    parse_mode: str = "abstract",
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
                id=uuid5(section_id, f"chunk:{idx}"),
                article_id=article_id,
                section_id=section_id,
                section_type=section_type,
                text=fragment,
                char_start=start,
                char_end=end,
                chunk_index=idx,
                token_count=len(fragment) // 4,   # estimation
                parse_mode=parse_mode,
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
    if article.source.acl_id:
        return f"https://aclanthology.org/{article.source.acl_id}.pdf"
    if article.source.arxiv_id:
        return f"https://arxiv.org/pdf/{article.source.arxiv_id}"
    if article.source.pdf_url:
        return str(article.source.pdf_url)
    return None


def _download_pdf(url: str, dest: Path) -> bool:
    """Telecharge un PDF dans dest. Retourne True si reussi."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(
            url,
            timeout=PDF_TIMEOUT,
            stream=True,
            headers={"User-Agent": "SciLit/0.2.0"},
        )
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
    t = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", title_line.strip())
    for pattern, stype in SECTION_PATTERNS:
        if pattern.match(t):
            return stype
    return SectionType.OTHER


def _reference_entries(lines: list[str]) -> list[str]:
    entries: list[str] = []
    current = ""
    numeric_start = re.compile(
        r"(?<!\S)(?:\[\d{1,3}\]|\d{1,3}[.)]?)[ \t]+(?=[A-Z])"
    )
    author_start = re.compile(r"^\s*[A-Z][A-Za-z’'’-]+,\s+")

    def append_piece(piece: str, starts_entry: bool = False) -> None:
        nonlocal current
        piece = piece.strip()
        if not piece:
            return
        if starts_entry or (
            current
            and author_start.match(piece)
            and re.search(r"\b(?:18|19|20)\d{2}[a-z]?\b", current, re.I)
        ):
            if current:
                entries.append(current.strip())
            current = piece
        elif current:
            current = f"{current} {piece}"
        else:
            current = piece

    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        matches = list(numeric_start.finditer(line))
        if not matches:
            append_piece(line)
            continue
        if matches[0].start():
            append_piece(line[:matches[0].start()])
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            append_piece(line[match.start():end], starts_entry=True)
    if current:
        entries.append(current.strip())
    return entries


def _extract_fulltext_sections(pdf_path: Path, article: Article) -> list[Section]:
    import pymupdf as fitz

    doc = fitz.open(str(pdf_path))
    lines: list[tuple[str, float, bool]] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if text:
                    lines.append((
                        text,
                        max((span.get("size", 0) for span in spans), default=0),
                        any(span.get("flags", 0) & 16 for span in spans),
                    ))
    doc.close()
    if not lines:
        return []
    sizes = sorted(size for _, size, _ in lines if size > 0)
    body_size = sizes[len(sizes) // 2] if sizes else 10.0
    sections: list[Section] = []
    ref_lines: list[str] = []
    section_type = SectionType.OTHER
    section_title: str | None = None
    text_lines: list[str] = []
    offset = section_index = 0
    in_references = False
    saw_body_start = False
    body_start_types = {
        SectionType.ABSTRACT,
        SectionType.INTRODUCTION,
        SectionType.RELATED_WORK,
        SectionType.METHOD,
        SectionType.EXPERIMENT,
        SectionType.RESULTS,
    }

    def heading_details(text: str, size: float, bold: bool):
        normalized = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text)
        kind = _detect_section_type(text)
        known = (
            kind != SectionType.OTHER
            and len(normalized.split()) <= 12
            and len(text) <= 100
            and not re.search(r"[.!?,;:]$", text)
            and (bold or size >= body_size * 1.05 or len(normalized.split()) <= 3)
        )
        numbered = (
            bool(re.match(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Za-z]\w*", text))
            and len(normalized.split()) <= 10
            and not re.search(r"[.!?]\s", normalized)
            and not re.search(r"[.!?,;:]$", text)
        )
        styled = (
            (size >= body_size * 1.25 or (bold and size >= body_size * 1.05))
            and len(text) <= 80
            and len(text.split()) <= 8
            and not re.search(r"[.!?,;:]$", text)
            and text[:1].isupper()
        )
        return normalized, kind, known or numbered, styled

    reference_heading_index = None
    prepass_body_start = False
    for line_index, (text, size, bold) in enumerate(lines):
        _, kind, structural, styled = heading_details(text, size, bold)
        if not prepass_body_start:
            if structural and kind in body_start_types:
                prepass_body_start = True
        elif (structural or styled) and kind == SectionType.REFERENCES:
            reference_heading_index = line_index

    def flush() -> None:
        nonlocal offset, section_index, text_lines
        text = " ".join(text_lines).strip()
        if text:
            end = offset + len(text)
            sections.append(Section(
                id=uuid5(article.id, f"section:{section_index}"),
                article_id=article.id,
                section_type=section_type,
                title=section_title,
                text=text,
                char_start=offset,
                char_end=end,
                section_index=section_index,
                parse_mode="fulltext",
            ))
            offset = end + 1
            section_index += 1
        text_lines = []

    for line_index, (text, size, bold) in enumerate(lines):
        normalized, kind, structural, styled = heading_details(text, size, bold)
        heading = structural or (saw_body_start and styled)
        if not saw_body_start:
            if structural and kind in body_start_types:
                saw_body_start = True
            else:
                continue
        if not saw_body_start:
            continue
        if heading and kind == SectionType.REFERENCES:
            if line_index != reference_heading_index:
                text_lines.append(text)
                continue
            flush()
            in_references = True
            section_title = normalized
            continue
        if in_references:
            ref_lines.append(text)
        elif heading:
            flush()
            section_type, section_title = kind, text
        else:
            text_lines.append(text)
    if not in_references:
        flush()
    entries = _reference_entries(ref_lines)
    if entries:
        text = "\n".join(entries)
        sections.append(Section(
            id=uuid5(article.id, f"section:{section_index}"),
            article_id=article.id,
            section_type=SectionType.REFERENCES,
            title="References",
            text=text,
            char_start=offset,
            char_end=offset + len(text),
            section_index=section_index,
            parse_mode="fulltext",
        ))
    return sections


def _extract_sections_pymupdf(pdf_path: Path, article: Article) -> list[Section]:
    """
    Extrait le texte du PDF avec PyMuPDF et segmente en sections par
    heuristiques (taille de police et patterns de titres).
    SciBERT remplacera ceci au palier Silver-avance (Lot C).
    """
    try:
        import pymupdf as fitz
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

    # Fallback : si aucune section detectee, abstract uniquement
    if not sections:
        abstract = article.abstract.strip()
        if abstract:
            sections.append(Section(
                id=uuid5(article.id, "section:0"),
                article_id=article.id,
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                text=abstract,
                char_start=0,
                char_end=len(abstract),
                section_index=0,
            ))

    return sections


def _is_redistributable(article: Article) -> bool:
    license_name = (article.license or "").casefold()
    return bool(article.source.acl_id) or license_name.startswith(
        ("cc-by", "cc0", "public-domain")
    )


def run_silver(
    articles: list[Article],
    max_articles: int = 300,
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
    section_rows = load_sections(sections_path)
    chunk_rows = load_chunks(chunks_path)
    section_done = {
        str(item.article_id) for item in section_rows
        if item.parse_mode == "fulltext"
    }
    chunk_done = {
        str(item.article_id) for item in chunk_rows
        if item.parse_mode == "fulltext"
    }
    completed = section_done & chunk_done
    selected = sorted(
        (a for a in articles if _is_redistributable(a)),
        key=lambda a: a.cited_by_count,
        reverse=True,
    )[:max_articles]
    total_s = total_c = downloaded = failed = 0
    pdf_dir = paths.raw.parent / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    last_arxiv_request: float | None = None
    log.info("Silver : traitement de %d articles licenciés.", len(selected))
    with (
        open(sections_path, "a", encoding="utf-8") as fs,
        open(chunks_path, "a", encoding="utf-8") as fc,
    ):
        for article in tqdm(selected, desc="Silver PDF", ncols=80):
            if str(article.id) in completed:
                continue
            pdf_url = _pdf_url_for(article)
            if not pdf_url:
                failed += 1
                continue
            is_arxiv = "arxiv.org/pdf/" in pdf_url
            pdf_path = pdf_dir / f"{article.corpus_id}.pdf"
            cached_pdf = pdf_path.exists() and pdf_path.stat().st_size > 1000
            if is_arxiv and not cached_pdf and last_arxiv_request is not None:
                time.sleep(max(0, 3.0 - (time.monotonic() - last_arxiv_request)))
            if not _download_pdf(pdf_url, pdf_path):
                failed += 1
                if is_arxiv:
                    last_arxiv_request = time.monotonic()
                continue
            if is_arxiv:
                last_arxiv_request = time.monotonic()
            downloaded += 1
            try:
                sections = _extract_fulltext_sections(pdf_path, article)
            except Exception as exc:
                log.warning("PDF parse failed for %s: %s", article.corpus_id, exc)
                failed += 1
                continue
            chunks = []
            for section in sections:
                chunks.extend(chunk_text(
                    section.text,
                    article.id,
                    section.id,
                    section.section_type,
                    parse_mode="fulltext",
                ))
            for section in sections:
                fs.write(section.model_dump_json() + "\n")
                total_s += 1
            for chunk in chunks:
                fc.write(chunk.model_dump_json(exclude={"embedding"}) + "\n")
                total_c += 1
            completed.add(str(article.id))
    log.info("Silver : PDF ok=%d echec=%d.", downloaded, failed)
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
    paths = config.paths()
    config.setup_logging(paths.parsed / "parse_log.txt")
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
        default=None,
        help=f"Chemin vers articles.jsonl (default: {paths.raw / 'articles.jsonl'})",
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
    print_stats(paths.parsed / "sections.jsonl", paths.parsed / "chunks.jsonl")

if __name__ == "__main__":
    main()
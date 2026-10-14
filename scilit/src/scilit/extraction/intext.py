from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from uuid import uuid5

from rapidfuzz import fuzz

from scilit import config
from scilit.corpus.parse import load_chunks, load_sections
from scilit.schemas import Article, Chunk, InTextCitation, Section, SectionType

YEAR = r"(?:18|19|20)\d{2}[a-z]?"
PAREN_AUTHOR_YEAR = re.compile(rf"\(([^()]{{1,180}}\b{YEAR}[^()]*)\)")
NARRATIVE_AUTHOR_YEAR = re.compile(
    rf"\b([A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?)\s*\(({YEAR})\)"
)
AUTHOR_NAMES = r"[A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'’-]+))*"
PAREN_AUTHOR_ITEM = re.compile(rf"^\s*({AUTHOR_NAMES})\s*,\s*({YEAR})\s*$")
PAREN_AUTHOR_ITEM_NO_COMMA = re.compile(
    rf"^\s*([A-Z][A-Za-z'’-]+\s+et\s+al\.)\s+({YEAR})\s*$"
)
NUMERIC_MARKER = re.compile(
    r"\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–]\s*\d+)?)*)\]"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\(\d{4})")


def _normalize_title(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _reference_title(reference: str) -> str:
    reference = re.sub(r"^\s*(?:\[\d+\]|\d{1,3}[.)]?)\s*", "", reference)
    match = re.search(rf"\b{YEAR}\.\s*(.+)", reference)
    if match:
        return match.group(1).strip()
    return reference


def _reference_for_author(
    entries: list[str], surname: str, year: str
) -> str | None:
    surname = surname.casefold()
    for entry in entries:
        entry_year = re.search(YEAR, entry)
        author_text = re.sub(
            r"^\s*(?:\[\d+\]|\d{1,3}[.)]?)\s*", "", entry
        )
        first_author = re.match(r"\s*([A-Z][A-Za-z'’-]+)", author_text)
        if (
            entry_year
            and entry_year.group().casefold() == year.casefold()
            and first_author
            and first_author.group(1).casefold() == surname
        ):
            return entry
    return None


def _target_article(
    reference: str | None,
    citing_article: Article,
    corpus: list[Article],
    by_openalex_id: dict[str, Article],
) -> Article | None:
    if not reference:
        return None
    title = _normalize_title(_reference_title(reference))
    if not title:
        return None
    preferred = [
        by_openalex_id[item]
        for item in citing_article.referenced_works
        if item in by_openalex_id
    ]
    for candidates in (preferred, corpus):
        scored = [
            (fuzz.token_set_ratio(title, _normalize_title(item.title)), item)
            for item in candidates
        ]
        score, article = max(scored, key=lambda pair: pair[0], default=(0, None))
        if article is not None and score >= 90:
            return article
    return None


def _marker_links(
    sentence: str,
    entries: list[str],
) -> list[tuple[str, str | None]]:
    links: list[tuple[str, str | None]] = []
    occupied: list[tuple[int, int]] = []
    for match in PAREN_AUTHOR_YEAR.finditer(sentence):
        marker = match.group(0)
        has_author_year = False
        for item in match.group(1).split(";"):
            author_year = (
                PAREN_AUTHOR_ITEM.match(item)
                or PAREN_AUTHOR_ITEM_NO_COMMA.match(item)
            )
            if author_year:
                has_author_year = True
                links.append((
                    marker,
                    _reference_for_author(
                        entries,
                        author_year.group(1).split()[0],
                        author_year.group(2),
                    ),
                ))
        if has_author_year:
            occupied.append(match.span())
    for match in NARRATIVE_AUTHOR_YEAR.finditer(sentence):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        marker = match.group(0)
        links.append((
            marker,
            _reference_for_author(entries, match.group(1).split()[0], match.group(2)),
        ))
    for match in NUMERIC_MARKER.finditer(sentence):
        indices: list[int] = []
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            bounds = re.split(r"\s*[-–]\s*", part)
            start = int(bounds[0])
            end = int(bounds[-1])
            indices.extend(range(start, min(end, len(entries)) + 1))
        links.extend(
            (match.group(0), entries[index - 1] if 1 <= index <= len(entries) else None)
            for index in indices
        )
    return links


def _reconstruct_section_text(chunks: list[Chunk]) -> str:
    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    if not ordered:
        return ""

    text = ordered[0].text
    previous = ordered[0]
    for chunk in ordered[1:]:
        overlap = previous.char_end - chunk.char_start
        if overlap <= 0:
            text += " " * max(0, chunk.char_start - previous.char_end)
            text += chunk.text
            previous = chunk
            continue

        expected = max(0, len(text) - overlap)
        radius = max(64, min(512, overlap))
        lower = max(0, expected - radius)
        upper = min(len(text), expected + radius + 1)
        matched = None
        for anchor_length in range(min(100, overlap, len(chunk.text)), 0, -1):
            anchor = chunk.text[:anchor_length]
            position = text.find(anchor, lower, upper)
            closest = None
            while position >= 0:
                overlap_length = len(text) - position
                if 0 < overlap_length <= len(chunk.text):
                    distance = abs(position - expected)
                    if closest is None or distance < closest[0]:
                        closest = (distance, overlap_length)
                position = text.find(anchor, position + 1, upper)
            if closest is not None:
                matched = closest[1]
                break

        if matched is None:
            matched = min(overlap, len(chunk.text))
        text += chunk.text[matched:]
        previous = chunk
    return text


def extract_intext_citations(
    sections: list[Section],
    articles: list[Article],
    chunks: list[Chunk] | None = None,
) -> list[InTextCitation]:
    by_article_id = {str(item.id): item for item in articles}
    by_openalex_id = {
        item.openalex_id: item for item in articles if item.openalex_id
    }
    section_texts = {str(item.id): item.text for item in sections}
    chunks_by_section: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks or []:
        chunks_by_section[str(chunk.section_id)].append(chunk)
    for section in sections:
        section_id = str(section.id)
        if not section_texts[section_id] and section_id in chunks_by_section:
            section_texts[section_id] = _reconstruct_section_text(
                chunks_by_section[section_id]
            )
    refs: dict[str, list[str]] = defaultdict(list)
    for section in sections:
        text = section_texts[str(section.id)]
        if (
            section.parse_mode == "fulltext"
            and section.section_type == SectionType.REFERENCES
        ):
            refs[str(section.article_id)] = [
                entry.strip() for entry in text.splitlines() if entry.strip()
            ]
    citations: list[InTextCitation] = []
    target_cache: dict[tuple[str, str | None], Article | None] = {}
    for section in sections:
        if section.parse_mode != "fulltext" or section.section_type == SectionType.REFERENCES:
            continue
        text = section_texts[str(section.id)]
        citing = by_article_id.get(str(section.article_id))
        if not citing:
            continue
        sentences = SENTENCE_SPLIT.split(text)
        for index, sentence in enumerate(sentences):
            for link_index, (marker, reference) in enumerate(
                _marker_links(sentence, refs[str(citing.id)])
            ):
                cache_key = (str(citing.id), reference)
                if cache_key not in target_cache:
                    target_cache[cache_key] = _target_article(
                        reference, citing, articles, by_openalex_id
                    )
                target = target_cache[cache_key]
                previous = sentences[index - 1] if index else ""
                following = sentences[index + 1] if index + 1 < len(sentences) else ""
                marker_index = text.find(marker)
                citations.append(InTextCitation(
                    id=uuid5(
                        section.id,
                        f"intext:{index}:{marker_index}:{marker}:{reference}:{link_index}",
                    ),
                    source_article_id=citing.id,
                    target_article_id=target.id if target else None,
                    target_raw_ref=marker,
                    marker_text=marker,
                    context_sentence=sentence.strip(),
                    context_window=" ".join(
                        item.strip() for item in (previous, sentence, following)
                        if item.strip()
                    ),
                    section_type=section.section_type,
                    reference_entry=reference,
                    reference_linked=reference is not None,
                    function="undefined",
                ))
    return citations


def save_intext_citations(
    sections: list[Section] | None = None,
    articles: list[Article] | None = None,
    output_path: Path | None = None,
    chunks: list[Chunk] | None = None,
) -> dict[str, float | int]:
    parsed = config.paths().parsed
    sections = sections if sections is not None else load_sections()
    chunks = chunks if chunks is not None else load_chunks()
    articles = articles if articles is not None else [
        Article.model_validate_json(line)
        for line in (config.paths().raw / "articles.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    citations = extract_intext_citations(sections, articles, chunks)
    output_path = output_path or parsed / "intext_citations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in citations),
        encoding="utf-8",
    )
    count = len(citations)
    refs_linked = sum(item.reference_linked for item in citations)
    articles_linked = sum(item.target_article_id is not None for item in citations)
    return {
        "markers": count,
        "reference_entry_linked_percent": 100 * refs_linked / count if count else 0.0,
        "corpus_article_linked_percent": (
            100 * articles_linked / count if count else 0.0
        ),
    }

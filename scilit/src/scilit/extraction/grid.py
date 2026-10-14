from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from uuid import UUID

from scilit.schemas import ComparativeCell, ComparativeRow

QUESTION_SENTENCE = re.compile(
    r"\b(?:we\s+(?:propose|present|introduce|study|investigate|explore)"
    r"|this\s+(?:paper|work))\b",
    re.I,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
DATASET_PATTERN = re.compile(
    r"\b(?:SQuAD(?:\s*2\.0)?|FEVER|GLUE|SuperGLUE|MMLU|WMT[- ]?\d{2,4}"
    r"|CoNLL[- ]?\d{4}|MS\s*MARCO|MNLI|SNLI|HotpotQA|TriviaQA)\b",
    re.I,
)


def _value(item, key: str, default=None):
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def _snippet(text: str) -> str:
    words = text.strip().split()
    if len(words) <= 25:
        return text.strip()
    return " ".join(words[:25])


def _cell(text: str, *, claim_id=None, chunk_id=None) -> ComparativeCell | None:
    text = _snippet(text)
    if not text or (claim_id is None and chunk_id is None):
        return None
    return ComparativeCell(text=text, claim_id=claim_id, chunk_id=chunk_id, verbatim=True)


def _claims_for(article_id: str, claims: Iterable) -> list:
    return [item for item in claims if str(_value(item, "article_id")) == article_id]


def _chunk_for_text(
    article_id: str,
    text: str,
    chunks: list,
    *,
    section_id: str | None = None,
    section_type: str | None = None,
) -> str | None:
    candidates = [
        chunk
        for chunk in chunks
        if str(_value(chunk, "article_id")) == article_id
        and (
            str(_value(chunk, "section_id")) == section_id
            if section_id is not None
            else str(_value(chunk, "section_type")) in (
                section_type,
                f"SectionType.{section_type.upper()}" if section_type else None,
            )
        )
    ]
    for chunk in candidates:
        if text in _value(chunk, "text", ""):
            return str(_value(chunk, "id"))
    return str(_value(candidates[0], "id")) if candidates else None


def _claim_cell(claim, text: str | None = None) -> ComparativeCell | None:
    return _cell(
        text or _value(claim, "verbatim") or _value(claim, "text", ""),
        claim_id=_value(claim, "id"),
    )


def build_comparison_grid(
    article_ids: Iterable[str | UUID],
    articles: Iterable | Mapping,
    claims: Iterable,
    sections: Iterable,
    chunks: Iterable,
) -> list[ComparativeRow]:
    article_map = (
        {str(key): value for key, value in articles.items()}
        if isinstance(articles, Mapping)
        else {str(_value(item, "id")): item for item in articles}
    )
    claims = list(claims)
    sections = list(sections)
    chunks = list(chunks)
    rows = []
    for raw_id in article_ids:
        article_id = str(raw_id)
        article = article_map.get(article_id)
        if article is None:
            continue
        article_claims = _claims_for(article_id, claims)
        abstract = _value(article, "abstract", "")
        abstract_sentences = SENTENCE_SPLIT.split(abstract.strip())
        question = next(
            (item.strip() for item in abstract_sentences if QUESTION_SENTENCE.search(item)),
            abstract_sentences[0].strip() if abstract_sentences else "",
        )
        abstract_chunk_id = _chunk_for_text(
            article_id, question, chunks, section_type="abstract"
        )
        question_cell = _cell(question, chunk_id=abstract_chunk_id)

        method = [
            cell for claim in article_claims
            if _value(claim, "claim_type") in ("method", "ClaimType.METHOD")
            if (cell := _claim_cell(claim)) is not None
        ]
        data_cells = []
        for claim in article_claims:
            claim_text = _value(claim, "verbatim") or _value(claim, "text", "")
            dataset = _value(claim, "dataset_name")
            if dataset:
                snippet = (
                    str(dataset)
                    if str(dataset).casefold() in claim_text.casefold()
                    else claim_text
                )
                cell = _claim_cell(claim, snippet)
            elif _value(claim, "claim_type") in ("dataset", "ClaimType.DATASET"):
                match = DATASET_PATTERN.search(claim_text)
                cell = _claim_cell(claim, match.group(0) if match else claim_text)
            else:
                match = DATASET_PATTERN.search(claim_text)
                cell = _claim_cell(claim, match.group(0)) if match else None
            if cell:
                data_cells.append(cell)

        results = [
            cell for claim in article_claims
            if _value(claim, "claim_type") in ("finding", "ClaimType.FINDING")
            and _value(claim, "metric_value") is not None
            if (cell := _claim_cell(claim)) is not None
        ]
        limitation_claims = [
            claim for claim in article_claims
            if _value(claim, "claim_type") in ("limitation", "ClaimType.LIMITATION")
        ]
        limitations = [
            cell for claim in limitation_claims
            if (cell := _claim_cell(claim)) is not None
        ]
        if not limitations:
            for section in sections:
                if not (
                    str(_value(section, "article_id")) == article_id
                    and _value(section, "section_type") in (
                        "limitations", "SectionType.LIMITATIONS"
                    )
                ):
                    continue
                section_text = _value(section, "text", "")
                section_id = str(_value(section, "id"))
                sources = (
                    [(section_text, None, section_id)]
                    if section_text
                    else [
                        (
                            _value(chunk, "text", ""),
                            _value(chunk, "id"),
                            section_id,
                        )
                        for chunk in chunks
                        if str(_value(chunk, "article_id")) == article_id
                        and str(_value(chunk, "section_id")) == section_id
                    ]
                )
                for source_text, source_chunk_id, source_section_id in sources:
                    for sentence in SENTENCE_SPLIT.split(source_text):
                        cell = _cell(
                            sentence,
                            chunk_id=source_chunk_id or _chunk_for_text(
                                article_id,
                                sentence.strip(),
                                chunks,
                                section_id=source_section_id,
                            ),
                        )
                        if cell:
                            limitations.append(cell)
        rows.append(ComparativeRow(
            article_id=UUID(article_id),
            question=[question_cell] if question_cell else [],
            method=method,
            data=data_cells,
            results=results,
            limitations=limitations,
        ))
    return rows

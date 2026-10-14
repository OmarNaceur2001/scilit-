from __future__ import annotations

from collections.abc import Mapping

from scilit.schemas import ReviewDraft, VerificationStatus

BADGES = {
    VerificationStatus.VERIFIED: "✓ verified",
    VerificationStatus.EXISTS_NO_SUPPORT: "~ no support",
    VerificationStatus.HALLUCINATED: "✗ hallucinated",
    VerificationStatus.UNVERIFIABLE: "? unverifiable",
}
STATUS_PRIORITY = (
    VerificationStatus.HALLUCINATED,
    VerificationStatus.EXISTS_NO_SUPPORT,
    VerificationStatus.UNVERIFIABLE,
    VerificationStatus.VERIFIED,
)


def _value(item, key: str, default=None):
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def render_review_markdown(
    draft: ReviewDraft,
    articles: Mapping[str, object] | None = None,
    include_badges: bool = True,
) -> str:
    articles = articles or {}
    numbers: dict[str, int] = {}
    for sentence in draft.sentences:
        for article_id in sentence.citations:
            key = str(article_id)
            if key not in numbers:
                numbers[key] = len(numbers) + 1

    output = [f"# {draft.title}", ""]
    for sentence in draft.sentences:
        if sentence.theme_heading:
            output.extend((f"## {sentence.theme_heading}", ""))
        line = sentence.text.strip()
        citation_numbers = [numbers[str(article_id)] for article_id in sentence.citations]
        if citation_numbers:
            line += " " + " ".join(f"[{number}]" for number in citation_numbers)
        if include_badges and sentence.verification_statuses:
            statuses = set(sentence.verification_statuses)
            status = next(
                (candidate for candidate in STATUS_PRIORITY if candidate in statuses),
                VerificationStatus.UNVERIFIABLE,
            )
            line += f" {BADGES[status]}"
        output.extend((line, ""))

    if numbers:
        output.extend(("## References", ""))
        for article_id, number in numbers.items():
            article = articles.get(article_id, {})
            title = _value(article, "title", f"Article {article_id}")
            authors = _value(article, "authors", [])
            year = _value(article, "year")
            doi = _value(article, "doi")
            citation = f"{', '.join(authors)}. " if authors else ""
            citation += f"({year}). " if year else ""
            citation += str(title)
            if doi:
                citation += f". https://doi.org/{doi}"
            output.append(f"[{number}] {citation}")
    return "\n".join(output).rstrip() + "\n"

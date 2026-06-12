from __future__ import annotations

from pathlib import Path

from .baselines import FrequencyDictionary


def spacy_available() -> bool:
    try:
        import spacy  # noqa: F401
    except Exception:
        return False
    return True


def analyze_with_spacy(text: str) -> list[dict]:
    import spacy

    nlp = spacy.load("ro_core_news_sm")
    doc = nlp(text)
    rows = []
    for token in doc:
        rows.append(
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "morph": str(token.morph),
                "dep": token.dep_,
            }
        )
    return rows


def build_linguistic_report(
    dictionary: FrequencyDictionary,
    hard_cases: list[tuple[str, str, str]],
    *,
    row_limit: int = 30,
) -> str:
    ambiguity = dictionary.ambiguity_inventory(min_count=2)[:row_limit]
    lines = [
        "# Linguistic Support Report",
        "",
        "spaCy is used here as an auxiliary analysis layer, not as the core model.",
        "RoWordNet/NLP-Cube hooks are left optional because they are not required for on-prem inference.",
        "",
        "## High-Ambiguity Forms From Corpus",
        "",
        "| base | total | forms | entropy |",
        "|---|---:|---|---:|",
    ]
    for row in ambiguity:
        forms = ", ".join(f"{form}:{count}" for form, count in row["forms"][:6])
        lines.append(f"| {row['base']} | {row['total']} | {forms} | {row['entropy']:.2f} |")

    lines.extend(["", "## spaCy Analysis For Hard Cases", ""])
    if not spacy_available():
        lines.append("spaCy is not installed in this environment.")
        return "\n".join(lines) + "\n"

    for source, expected, why in hard_cases:
        lines.extend([f"### `{source}`", "", f"Expected: `{expected}`", "", why, ""])
        rows = analyze_with_spacy(expected)
        lines.extend(["| token | lemma | pos | morph | dep |", "|---|---|---|---|---|"])
        for row in rows:
            morph = row["morph"].replace("|", "\\|")
            lines.append(f"| {row['text']} | {row['lemma']} | {row['pos']} | {morph} | {row['dep']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_linguistic_report(
    dictionary_path: str | Path,
    hard_cases: list[tuple[str, str, str]],
    out_path: str | Path,
) -> None:
    dictionary = FrequencyDictionary.load(dictionary_path)
    report = build_linguistic_report(dictionary, hard_cases)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


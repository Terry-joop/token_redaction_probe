import argparse
import re

from common import write_jsonl
from medical_common import read_records, word_offsets


SENSITIVE_TUIS = {
    "T047", "T048", "T191", "T046", "T184", "T037", "T019", "T050",
    "T121", "T200", "T109", "T195", "T061", "T060", "T059", "T034",
    "T033", "T023", "T116", "T123", "T005", "T007", "T004", "T204",
}
PII_KEEP = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "MONEY", "PERCENT"}
CLINICAL_STOP = {
    "diagnosis", "diagnoses", "procedure", "procedures", "history", "impression",
    "indication", "indications", "findings", "finding", "examination", "exam",
    "technique", "comparison", "assessment", "plan", "description", "preoperative",
    "postoperative", "operative", "disposition", "complaints", "complaint",
    "consultation", "reason", "left", "right", "bilateral", "lateral", "medial",
    "anterior", "posterior", "superior", "inferior", "upper", "lower", "negative",
    "positive", "normal", "abnormal", "general", "present", "absent", "stable",
    "unremarkable", "gross", "patient", "male", "female", "estimated", "performed",
    "well", "without",
}
DURATION = re.compile(
    r"^\s*(about |around |~)?\d+\s*(day|week|month|year|hour|min|yr|mo|wk)s?\b", re.I
)


def load_pipelines(threshold: float):
    try:
        import spacy
        import scispacy  # noqa: F401 - registers scispaCy pipeline components
        from scispacy.linking import EntityLinker  # noqa: F401 - registers linker factory
    except ImportError as error:
        raise SystemExit(
            "medical NER dependencies are missing. Install requirements-medical.txt "
            "and the en_core_sci_sm/en_core_web_sm models first."
        ) from error
    try:
        science = spacy.load("en_core_sci_sm", disable=["parser", "lemmatizer"])
        science.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True, "linker_name": "umls", "threshold": threshold,
        })
        pii = spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"]
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"could not load medterm4 models/linker: {error}") from error
    return science, science.get_pipe("scispacy_linker"), pii


def keep_word(word: str) -> bool:
    from spacy.lang.en.stop_words import STOP_WORDS

    normalized = word.strip().lower()
    lexical = normalized.strip(".,:;!?()[]{}\"'`-/\\")
    if (
        not normalized
        or normalized in STOP_WORDS
        or normalized in CLINICAL_STOP
        or lexical in CLINICAL_STOP
    ):
        return False
    if not any(character.isalpha() or character.isdigit() for character in normalized):
        return False
    return not (len(normalized) == 1 and normalized.isalpha())


def annotate(text: str, words: list[str], science, linker, pii) -> tuple[list[int], list[str]]:
    offset_words, offsets = word_offsets(text)
    if offset_words != words:
        raise ValueError("stored words do not match tokenizer offsets")
    spans = []
    for entity in science(text).ents:
        if not entity._.kb_ents:
            continue
        concept = linker.kb.cui_to_entity[entity._.kb_ents[0][0]]
        if set(concept.types) & SENSITIVE_TUIS:
            spans.append((entity.start_char, entity.end_char, "MEDICAL"))
    for entity in pii(text).ents:
        entity_type = entity.label_
        if entity_type in PII_KEEP:
            normalized = "LOCATION" if entity_type in {"GPE", "LOC", "FAC"} else entity_type
            spans.append((entity.start_char, entity.end_char, normalized))
        elif entity_type == "DATE" and re.search(r"\b(19|20)\d\d\b", entity.text):
            if not DURATION.match(entity.text):
                spans.append((entity.start_char, entity.end_char, "DATE"))

    labels = [0] * len(words)
    types = ["O"] * len(words)
    for index, (start, end) in enumerate(offsets):
        if not keep_word(words[index]):
            continue
        for span_start, span_end, span_type in spans:
            if start < span_end and span_start < end:
                labels[index] = 1
                types[index] = span_type
                break
    return labels, types


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate raw words with the medterm4 policy")
    parser.add_argument("--input", default="data/medical_redactor/drugreviews/pilot_input.jsonl")
    parser.add_argument("--output", default="data/medical_redactor/drugreviews/medterm4.jsonl")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    science, linker, pii = load_pipelines(args.threshold)
    output = []
    rows = read_records(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    for number, row in enumerate(rows, start=1):
        labels, types = annotate(row["text"], row["words"], science, linker, pii)
        output.append({
            "id": row["id"], "labels": labels, "types": types,
            "selected_words": [word for word, label in zip(row["words"], labels) if label],
            "source": "medterm4-reimplementation-v2-latest-aligned",
        })
        if number % 25 == 0:
            print(f"annotated {number}/{len(rows)}")
    write_jsonl(args.output, output)
    total = sum(len(row["words"]) for row in rows)
    selected = sum(sum(row["labels"]) for row in output)
    print(f"wrote {len(output)} rows to {args.output}; mask_rate={selected / max(total, 1):.2%}")


if __name__ == "__main__":
    main()

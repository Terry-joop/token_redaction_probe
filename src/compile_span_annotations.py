import argparse

from common import write_jsonl
from medical_common import read_records


def find_all(words: list[str], phrase: list[str]) -> list[tuple[int, int]]:
    width = len(phrase)
    return [
        (start, start + width)
        for start in range(len(words) - width + 1)
        if words[start:start + width] == phrase
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile exact selected word spans to labels")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_records(args.inputs)}
    output = []
    for annotation in read_records(args.spans):
        source = inputs[annotation["id"]]
        labels = [0] * len(source["words"])
        types = ["O"] * len(source["words"])
        for selected in annotation["selected_spans"]:
            phrase = selected["words"]
            matches = find_all(source["words"], phrase)
            if not matches:
                raise ValueError(f"{annotation['id']}: phrase not found: {phrase}")
            requested = selected.get("occurrence", "all")
            if requested != "all":
                index = int(requested)
                if index < 0 or index >= len(matches):
                    raise ValueError(f"{annotation['id']}: occurrence out of range: {phrase}")
                matches = [matches[index]]
            for start, end in matches:
                for position in range(start, end):
                    labels[position] = 1
                    types[position] = selected["type"]
        output.append({
            "id": source["id"], "labels": labels, "types": types,
            "selected_words": [word for word, label in zip(source["words"], labels) if label],
            "source": annotation.get("source", "span-teacher"),
            "needs_review": bool(annotation.get("needs_review", False)),
            "review_reason": annotation.get("review_reason", ""),
        })
    write_jsonl(args.output, output)
    print(f"compiled {len(output)} annotations to {args.output}")


if __name__ == "__main__":
    main()

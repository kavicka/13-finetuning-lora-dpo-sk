"""Error analysis of the fine-tuned model on the test set: where do the remaining mistakes sit?"""

import json
from collections import Counter

from common import load_split
from schema import FIELDS, TEST_ONLY_CITIES, norm, parse_json, score_fields, stem


def load_preds(name: str) -> list[dict]:
    return [json.loads(line) for line in open(f"data/preds_{name}.jsonl")]


def main() -> None:
    preds = load_preds("ollama_finetuned")
    zero = {p["index"]: p for p in load_preds("zero_shot")}
    text = {r["index"]: r["text"] for r in load_split("test")}
    errors, examples = Counter(), {f: [] for f in FIELDS}
    city = {"seen_ok": 0, "seen_bad": 0, "new_ok": 0, "new_bad": 0, "inflected_answer": 0, "invented_for_null": 0}
    claim = Counter()
    perfect = 0
    for p in preds:
        out, rec = parse_json(p["raw"]) or {}, p["record"]
        counts = score_fields(out, rec)
        wrong = [f for f in FIELDS if counts[f][1] or counts[f][2]]
        perfect += not wrong
        errors.update(wrong)
        for f in wrong:
            if len(examples[f]) < 6:
                examples[f].append({"gold": rec[f], "pred": out.get(f)})
        if rec["city"] is None:
            city["invented_for_null"] += out.get("city") is not None
        else:
            good = norm(out.get("city") or "") == norm(rec["city"])
            key = "new" if rec["city"] in TEST_ONLY_CITIES else "seen"
            city[f"{key}_{'ok' if good else 'bad'}"] += 1
            city["inflected_answer"] += (not good) and stem(out.get("city") or "", 3) == stem(rec["city"], 3)
        if "claim_type" in wrong:
            claim[f"{rec['claim_type']} → {out.get('claim_type')}"] += 1
    disagree = sum(
        1
        for p in preds
        if "city"
        in [f for f, c in score_fields(parse_json(zero[p["index"]]["raw"]), p["record"]).items() if c[1] or c[2]]
    )
    result = {
        "documents": len(preds),
        "documents_all_fields_right": perfect,
        "documents_with_error_by_field": dict(errors),
        "city": city,
        "zero_shot_city_errors": disagree,
        "claim_type_confusions": claim.most_common(8),
        "examples": examples,
        "answers_on_one_line": {
            "fine_tuned": sum("\n" not in p["raw"].strip() for p in preds),
            "zero_shot": sum("\n" not in p["raw"].strip() for p in zero.values()),
        },
        "answer_chars_mean": {
            "fine_tuned": sum(len(p["raw"]) for p in preds) / len(preds),
            "zero_shot": sum(len(p["raw"]) for p in zero.values()) / len(zero),
        },
        "report_chars_mean": sum(len(t) for t in text.values()) / len(text),
    }
    json.dump(result, open("results_error_analysis.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False)[:1800])


if __name__ == "__main__":
    main()

"""Catastrophic forgetting, the cheap way: does the tuned model still answer plain Slovak questions?

No system prompt on purpose. Each question has a keyword the right answer must contain. "as_json" counts answers
where the tuned model falls back to its one trick, which is the failure this check is really looking for.
"""

import argparse
import json

from eval_ollama import STUDENT, chat
from schema import norm, parse_json

QUESTIONS = [
    ("Aké je hlavné mesto Slovenska?", "bratislav"),
    ("Koľko je 7 krát 8?", "56"),
    ("Aká je chemická značka zlata?", "au"),
    ("Kto napísal Hamleta?", "shakespear"),
    ("Na ktorom kontinente leží Egypt?", "afri"),
    ("Prelož do angličtiny slovo pes.", "dog"),
    ("Aká je najvyššia hora Slovenska?", "gerlach"),
    ("Koľko dní má priestupný rok?", "366"),
]


def ask(model: str) -> list[dict]:
    rows = []
    for question, key in QUESTIONS:
        answer, _ = chat(model, [{"role": "user", "content": question}])
        rows.append(
            {
                "question": question,
                "answer": answer[:300],
                "correct": key in norm(answer),
                "as_json": parse_json(answer) is not None,
            }
        )
    return rows


def main(tuned: str) -> None:
    base, after = ask(STUDENT), ask(tuned)
    summary = {
        "questions": len(QUESTIONS),
        "base_correct": sum(r["correct"] for r in base),
        "tuned_correct": sum(r["correct"] for r in after),
        "base_as_json": sum(r["as_json"] for r in base),
        "tuned_as_json": sum(r["as_json"] for r in after),
        "base": base,
        "tuned": after,
    }
    json.dump(summary, open("results_forgetting.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in summary.items() if k not in ("base", "tuned")})
    for b, t in zip(base, after):
        print(
            b["question"],
            "| base:",
            b["answer"][:70].replace("\n", " "),
            "| tuned:",
            t["answer"][:70].replace("\n", " "),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claims-sk")
    main(ap.parse_args().model)

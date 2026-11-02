"""Synthetic claim reports with ground truth by construction.

The record is sampled first, then a local model writes a report containing exactly those facts, so the label
cannot be wrong unless the writer disobeys, and a consistency check catches most of that. Test records are written
by a different model than train and val, so the student cannot score by recognising one writer's style.

Appends to data/*.jsonl as it goes, so a stopped run resumes where it ended.
"""

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

from schema import STYLES, TEST_ONLY_CITIES, TRAIN_CITIES, consistency_problems, sample_record

# n is what gets written; about a quarter fail the consistency check, which leaves roughly 800 / 100 / 150
SPLITS = {
    "train": {"n": 1040, "seed": 1000, "writer": "qwen3:30b-a3b-instruct-2507-q4_K_M", "cities": TRAIN_CITIES},
    "val": {"n": 140, "seed": 2000, "writer": "qwen3:30b-a3b-instruct-2507-q4_K_M", "cities": TRAIN_CITIES},
    "test": {"n": 260, "seed": 3000, "writer": "gemma3:12b", "cities": TRAIN_CITIES + TEST_ONLY_CITIES},
}
TEMPERATURE = 0.8


def facts_for_writer(rec: dict) -> str:
    lines = [f"typ udalosti: {rec['claim_type']}"]
    if rec["event_date"]:
        lines.append(
            f"dátum udalosti: {rec['event_date']} (napíš ho po slovensky, napríklad 3. 5. 2024 alebo 3. mája 2024)"
        )
    if rec["city"]:
        lines.append(f"mesto: {rec['city']}")
    if rec["policy_number"]:
        lines.append(f"číslo poistky: {rec['policy_number']} (presne takto)")
    lines.append("poškodené veci: " + ", ".join(rec["damaged_items"]))
    if rec["estimated_amount_eur"] is not None:
        lines.append(f"odhad škody: {rec['estimated_amount_eur']} eur")
    lines.append("niekto bol zranený" if rec["injured_persons"] else "nikto nebol zranený (môžeš to aj vynechať)")
    lines.append(
        "udalosť bola nahlásená polícii"
        if rec["police_reported"]
        else "polícia nebola privolaná (môžeš to aj vynechať)"
    )
    return "\n".join(lines)


def writer_prompt(rec: dict, style: str, words: int) -> str:
    missing = [name for name, key in (("dátum", "event_date"), ("mesto", "city"), ("číslo poistky", "policy_number"), ("sumu škody", "estimated_amount_eur")) if rec[key] is None]  # fmt: skip
    rule = (
        f"Klient nepozná: {', '.join(missing)}. Tieto údaje v texte vôbec neuvádzaj, ani ich nevymýšľaj.\n"
        if missing
        else ""
    )
    return (
        f"Napíš po slovensky hlásenie poistnej udalosti od klienta. Štýl: {style}. Dĺžka asi {words} slov.\n"
        f"Text musí obsahovať presne tieto fakty a žiadne iné čísla, dátumy ani mestá:\n{facts_for_writer(rec)}\n{rule}"
        "Skloňuj slová prirodzene, ale názvy poškodených vecí a mesto musia byť v texte. Vymysli vierohodné okolnosti. "
        "Nepoužívaj zátvorky so zástupnými textami ako [Meno]; klient sa pokojne nepodpíše. Vypíš iba text hlásenia."
    )


def write(model: str, prompt: str, seed: int) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "keep_alive": "10m", "options": {"temperature": TEMPERATURE, "seed": seed, "num_predict": 400}}  # fmt: skip
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"].strip()


def main(limit: int | None) -> None:
    Path("data").mkdir(exist_ok=True)
    stats = {}
    for name, cfg in SPLITS.items():
        path = Path(f"data/{name}.jsonl")
        rows = [json.loads(line) for line in open(path)] if path.exists() else []
        done = {r["index"] for r in rows}
        rng = random.Random(cfg["seed"])
        # draw every record up front, so resuming and the split itself do not depend on what the writer did
        drawn = [(sample_record(rng, cfg["cities"]), rng.choice(STYLES), rng.randint(40, 120)) for _ in range(cfg["n"])]
        with open(path, "a") as out:
            for i, (rec, style, words) in enumerate(drawn[: limit or cfg["n"]]):
                if i in done:
                    continue
                text = write(cfg["writer"], writer_prompt(rec, style, words), cfg["seed"] + i)
                text = re.sub(r"\n{3,}", "\n\n", text)
                row = {
                    "index": i,
                    "record": rec,
                    "style": style,
                    "text": text,
                    "writer": cfg["writer"],
                    "problems": consistency_problems(text, rec),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                if i % 25 == 0:
                    print(name, i, "failed so far", sum(bool(r["problems"]) for r in rows), flush=True)
        for r in rows:  # the check improved after generation, so verdicts stored in the file may be old
            r["problems"] = consistency_problems(r["text"], r["record"])
        kinds = Counter(p.split(":")[0] for r in rows for p in r["problems"])
        stats[name] = {"generated": len(rows), "failed_check": sum(bool(r["problems"]) for r in rows), "kept": sum(not r["problems"] for r in rows), "problem_kinds": dict(kinds), "writer": cfg["writer"], "seed": cfg["seed"]}  # fmt: skip
        print(name, stats[name], flush=True)
    stats["temperature"] = TEMPERATURE
    json.dump(stats, open("results_data.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N of each split, for a trial")
    main(ap.parse_args().limit)

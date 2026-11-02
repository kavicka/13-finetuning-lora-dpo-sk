"""What every system shares: the prompt, the kept data, and the scoring of a predictions file."""

import json
import random
from pathlib import Path

import numpy as np

from schema import CLAIM_TYPES, FIELDS, consistency_problems, f1, parse_json, score_fields, valid_schema

BOOTSTRAP_SEED = 0
SYSTEM = (
    "Si systém poisťovne, ktorý z hlásenia poistnej udalosti vyberá údaje. Odpovedz iba jedným JSON objektom bez "
    "ďalšieho textu a bez markdownu. Kľúče:\n"
    f"claim_type: jedna z hodnôt {', '.join(CLAIM_TYPES)}\n"
    "event_date: dátum udalosti vo formáte YYYY-MM-DD, alebo null ak v texte nie je úplný dátum\n"
    "city: mesto v 1. páde, alebo null\n"
    "policy_number: číslo poistky presne ako v texte, alebo null\n"
    "damaged_items: zoznam poškodených vecí, krátke podstatné mená v 1. páde\n"
    "estimated_amount_eur: odhad škody v eurách ako číslo, alebo null\n"
    "injured_persons: true ak bol niekto zranený, inak false\n"
    "police_reported: true ak bola udalosť nahlásená polícii, inak false"
)


SPLIT_SIZES = {"train": 800, "val": 100, "test": 150}


def load_split(name: str) -> list[dict]:
    """The first reports that pass the consistency check, re-run here so the file never holds a stale verdict."""
    rows = [json.loads(line) for line in open(f"data/{name}.jsonl")]
    return [r for r in rows if not consistency_problems(r["text"], r["record"])][: SPLIT_SIZES[name]]


def gold_json(rec: dict) -> str:
    """One canonical key order: the student should not spend weights on formatting noise."""
    return json.dumps({f: rec[f] for f in FIELDS}, ensure_ascii=False)


def messages(text: str, shots: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots or []:
        msgs += [{"role": "user", "content": s["text"]}, {"role": "assistant", "content": gold_json(s["record"])}]
    return msgs + [{"role": "user", "content": text}]


def few_shots(train: list[dict], k: int = 3) -> list[dict]:
    """Fixed by seed before looking at any result; at most one per claim type, so the examples show variety."""
    rng, seen, shots = random.Random(7), set(), []
    for r in rng.sample(train, len(train)):
        t = r["record"]["claim_type"]
        if t not in seen:
            seen.add(t)
            shots.append(r)
        if len(shots) == k:
            break
    return shots


def micro(counts: list[dict]) -> float:
    tp, fp, fn = np.sum([[c[f] for f in FIELDS] for c in counts], axis=(0, 1))
    return f1(int(tp), int(fp), int(fn))["f1"]


def summarise(preds: list[dict]) -> dict:
    """preds: [{"record", "raw", "seconds"}]. Micro F1 over all fields, per-field F1, and a bootstrap over documents."""
    parsed = [parse_json(p["raw"]) for p in preds]
    counts = [score_fields(o, p["record"]) for o, p in zip(parsed, preds)]
    per_field = {f: f1(*np.sum([c[f] for c in counts], axis=0).tolist()) for f in FIELDS}
    tp, fp, fn = np.sum([[c[f] for f in FIELDS] for c in counts], axis=(0, 1)).tolist()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = [micro([counts[i] for i in rng.integers(0, len(counts), len(counts))]) for _ in range(1000)]
    return {
        "documents": len(preds),
        "valid_json_pct": 100 * float(np.mean([valid_schema(o) for o in parsed])),
        "parses_pct": 100 * float(np.mean([o is not None for o in parsed])),
        "micro": f1(tp, fp, fn),
        "micro_f1_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "per_field": per_field,
        "median_latency_s": float(np.median([p["seconds"] for p in preds])),
    }


def save_eval(name: str, preds: list[dict], meta: dict) -> dict:
    Path("data").mkdir(exist_ok=True)
    with open(f"data/preds_{name}.jsonl", "w") as out:
        out.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
    summary = {"name": name, **meta, **summarise(preds)}
    json.dump(summary, open(f"results_eval_{name}.json", "w"), indent=1, ensure_ascii=False)
    print(
        name,
        {k: summary[k] for k in ("documents", "valid_json_pct", "median_latency_s")},
        "micro F1",
        round(summary["micro"]["f1"], 4),
        summary["micro_f1_ci95"],
        flush=True,
    )
    return summary

"""Turn results_*.json into the figures in assets/. No measuring here."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from schema import FIELDS, parse_json, score_fields

SYSTEMS = [
    ("zero_shot", "zero-shot", "#94a3b8"),
    ("few_shot", "few-shot (3)", "#f97316"),
    ("lora", "LoRA SFT", "#2563eb"),
    ("lora_dpo", "LoRA + DPO", "#7c3aed"),
    ("ollama_finetuned", "fine-tuned in Ollama", "#059669"),
]


def available() -> list[tuple[str, str, str, dict]]:
    out = []
    for key, label, color in SYSTEMS:
        path = Path(f"results_eval_{key}.json")
        if path.exists():
            out.append((key, label, color, json.load(open(path))))
    return out


def per_field(systems) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    width = 0.8 / len(systems)
    x = np.arange(len(FIELDS))
    for i, (_, label, color, res) in enumerate(systems):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            [res["per_field"][f]["f1"] for f in FIELDS],
            width,
            label=f"{label} (micro {res['micro']['f1']:.3f})",
            color=color,
        )
    ax.set_xticks(x, [f.replace("_", "\n") for f in FIELDS], fontsize=9)
    ax.set(ylabel="field F1", ylim=(0, 1.04), title="Field-level F1 on the 150 held-out reports")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig("assets/per_field_f1.png", dpi=150)


def learning_curve() -> None:
    runs = json.load(open("results_learning_curve.json"))["runs"]
    full = json.load(open("results_eval_lora.json"))
    sizes = sorted(int(n) for n in runs)
    points = [(n, runs[str(n)]) for n in sizes] + [(800, full)]
    xs = [n for n, _ in points]
    ys = [r["micro"]["f1"] for _, r in points]
    low = [y - r["micro_f1_ci95"][0] for y, (_, r) in zip(ys, points)]
    high = [r["micro_f1_ci95"][1] - y for y, (_, r) in zip(ys, points)]
    few = json.load(open("results_eval_few_shot.json"))["micro"]["f1"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(xs, ys, yerr=[low, high], fmt="-o", color="#2563eb", capsize=3, label="LoRA, 95 % interval")
    ax.axhline(few, color="#f97316", ls="--", label="few-shot prompt")
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set(xlabel="training reports", ylabel="micro F1 on the test set", title="How many examples does this task need?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/learning_curve.png", dpi=150)


def wrong_fields(raw: str, rec: dict) -> list[str]:
    counts = score_fields(parse_json(raw), rec)
    return [f for f in FIELDS if counts[f][1] or counts[f][2]]


def before_after() -> None:
    """The first test report with at least 3 zero-shot errors that the tuned model gets fully right: fixed rule, no browsing."""
    best = "lora_dpo" if json.load(open("results_best.json"))["best"] == "lora_dpo" else "lora"
    zero = [json.loads(line) for line in open("data/preds_zero_shot.jsonl")]
    tuned = {r["index"]: r for r in map(json.loads, open(f"data/preds_{best}.jsonl"))}
    row = next(
        z
        for z in zero
        if len(wrong_fields(z["raw"], z["record"])) >= 3 and not wrong_fields(tuned[z["index"]]["raw"], z["record"])
    )
    text = tuned[row["index"]]["text"]
    bad = wrong_fields(row["raw"], row["record"])
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        (a0, "Report (test set, written by Gemma)", textwrap.fill(" ".join(text.split())[:620], 48), "#f8fafc"),
        (
            a1,
            f"Zero-shot: wrong = {', '.join(bad)}",
            json.dumps(parse_json(row["raw"]), ensure_ascii=False, indent=1),
            "#fef2f2",
        ),
        (
            a2,
            "Fine-tuned: every field right",
            json.dumps(parse_json(tuned[row["index"]]["raw"]), ensure_ascii=False, indent=1),
            "#f0fdf4",
        ),
    ]
    for ax, title, body, color in panels:
        ax.set_facecolor(color)
        ax.set(xticks=[], yticks=[])
        ax.set_title(textwrap.fill(title, 44), fontsize=10)
        ax.text(0.03, 0.97, body, va="top", family="monospace", fontsize=8.5, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig("assets/before_after.png", dpi=150)


def loss_curve() -> None:
    c = json.load(open("results_train_sft.json"))["curve"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([p["iter"] for p in c["train"]], [p["loss"] for p in c["train"]], color="#93c5fd", label="train")
    ax.plot(
        [p["iter"] for p in c["val"]], [p["loss"] for p in c["val"]], "-o", ms=3, color="#2563eb", label="validation"
    )
    ax.set(
        xlabel="iteration (batch 4)",
        ylabel="loss per answer token",
        yscale="log",
        title="LoRA SFT: most of the learning happens in the first 50 steps",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/loss_curve.png", dpi=150)


if __name__ == "__main__":
    Path("assets").mkdir(exist_ok=True)
    systems = available()
    per_field(systems)
    loss_curve()
    before_after()
    learning_curve()

"""How many examples does this task need? The same LoRA recipe on the first N training reports.

Every run gets the same number of steps, so a small set is seen many more times: this asks what the data alone
buys, not what one fixed number of epochs buys. Resumable: each size writes its own result. No model is chosen here.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from mlx_lm import load

from common import gold_json, load_split, messages, summarise
from eval_mlx import run
from train_config import BASE_MODEL, LORA

SIZES = (25, 100, 400)
ITERS = 300


def train(n: int, rows: list[dict]) -> str:
    folder, adapter = f"data/mlx_n{n}", f"adapters/n{n}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/train.jsonl", "w") as out:
        out.writelines(
            json.dumps(
                {"messages": messages(r["text"]) + [{"role": "assistant", "content": gold_json(r["record"])}]},
                ensure_ascii=False,
            )
            + "\n"
            for r in rows[:n]
        )
    shutil.copy(f"{LORA['data']}/valid.jsonl", f"{folder}/valid.jsonl")
    cfg = {
        **LORA,
        "data": folder,
        "adapter_path": adapter,
        "iters": ITERS,
        "steps_per_eval": ITERS,
        "val_batches": 5,
        "save_every": ITERS,
    }
    yaml.safe_dump(cfg, open(f"logs/lora_n{n}.yaml", "w"))
    subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--config", f"logs/lora_n{n}.yaml"],
        check=True,
        stdout=open(f"logs/train_n{n}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    return adapter


def main() -> None:
    rows, test = load_split("train"), load_split("test")
    results = (
        json.load(open("results_learning_curve.json"))
        if Path("results_learning_curve.json").exists()
        else {"iters": ITERS, "runs": {}}
    )
    for n in SIZES:
        if str(n) in results["runs"]:
            continue
        adapter = train(n, rows)
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
        s = summarise(run(model, tokenizer, test, f"n{n}"))
        results["runs"][str(n)] = {k: s[k] for k in ("valid_json_pct", "micro", "micro_f1_ci95", "per_field")}
        json.dump(results, open("results_learning_curve.json", "w"), indent=1)
        print(n, round(s["micro"]["f1"], 4), s["valid_json_pct"], flush=True)
        shutil.rmtree(adapter)  # only the number is kept; disk is tight


if __name__ == "__main__":
    main()

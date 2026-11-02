"""LoRA with plain supervised fine-tuning, through `mlx_lm.lora`. Writes the data, the config and the loss curve."""

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from common import gold_json, load_split, messages
from train_config import LORA


def write_data() -> dict:
    """mask_prompt then trains only on the answer: the report itself is context, not something to reproduce."""
    Path(LORA["data"]).mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, target in (("train", "train"), ("val", "valid")):
        rows = load_split(split)
        with open(f"{LORA['data']}/{target}.jsonl", "w") as out:
            for r in rows:
                msgs = messages(r["text"]) + [{"role": "assistant", "content": gold_json(r["record"])}]
                out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        sizes[split] = len(rows)
    return sizes


def parse_log(log: str) -> dict:
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)  # the progress table is coloured
    train = [
        {"iter": int(i), "loss": float(x)} for i, x in re.findall(r"^\s*(\d+)\s+([\d.]+) [▼▲·\-=]?", log, re.MULTILINE)
    ]
    val = [{"iter": int(i), "loss": float(x)} for i, x in re.findall(r"^\s*(\d+)\s+val ([\d.]+)", log, re.MULTILINE)]
    return {"train": train, "val": val}


def main() -> None:
    sizes = write_data()
    Path("logs").mkdir(exist_ok=True)
    yaml.safe_dump(LORA, open("logs/lora.yaml", "w"))
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "mlx_lm", "lora", "--config", "logs/lora.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if proc.wait():
        raise SystemExit(proc.returncode)
    json.dump(
        {"config": LORA, "examples": sizes, "curve": parse_log("".join(lines))},
        open("results_train_sft.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()

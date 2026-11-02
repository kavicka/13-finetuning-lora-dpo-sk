"""Run the student with an adapter in MLX over a split, with the same prompt and scoring as the Ollama systems."""

import argparse
import time

from mlx_lm import generate, load

from common import load_split, messages, save_eval
from train_config import BASE_MODEL

MAX_TOKENS = 300


def run(model, tokenizer, rows: list[dict], label: str) -> list[dict]:
    preds = []
    for i, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            messages(row["text"]), add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        start = time.perf_counter()
        raw = generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS)  # greedy: temperature 0
        preds.append(
            {
                "index": row["index"],
                "record": row["record"],
                "raw": raw,
                "seconds": time.perf_counter() - start,
                "text": row["text"],
            }
        )
        if i % 50 == 0:
            print(label, i, round(preds[-1]["seconds"], 2), flush=True)
    return preds


def main(name: str, adapter: str | None, split: str) -> None:
    model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
    preds = run(model, tokenizer, load_split(split), name)
    save_eval(
        name,
        preds,
        {
            "model": BASE_MODEL,
            "adapter": adapter,
            "split": split,
            "runtime": "mlx",
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test", help="val chooses the adapter, so the test set never does")
    a = ap.parse_args()
    main(a.name, a.adapter, a.split)

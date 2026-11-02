"""Run a model served by Ollama over the test set: zero-shot, few-shot, or the fused fine-tuned model."""

import argparse
import json
import time
import urllib.request

from common import few_shots, load_split, messages, save_eval

STUDENT = "qwen3:1.7b"
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 300}


def chat(model: str, msgs: list[dict]) -> tuple[str, float]:
    body = {"model": model, "messages": msgs, "stream": False, "think": False, "keep_alive": "10m", "options": OPTIONS}
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    answer = json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"]
    return answer, time.perf_counter() - start


def main(name: str, model: str, k: int) -> None:
    shots = few_shots(load_split("train"), k) if k else None
    test = load_split("test")
    chat(model, messages(test[0]["text"], shots))  # loads the model, so the first document is not charged for it
    preds = []
    for i, row in enumerate(test):
        raw, seconds = chat(model, messages(row["text"], shots))
        preds.append({"index": row["index"], "record": row["record"], "raw": raw, "seconds": seconds})
        if i % 50 == 0:
            print(name, i, round(seconds, 2), flush=True)
    meta = {
        "model": model,
        "runtime": "ollama",
        "shots": k,
        "few_shot_indices": [s["index"] for s in shots or []],
        "options": OPTIONS,
        "think": False,
    }
    save_eval(name, preds, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args()
    main(a.name, a.model, a.shots)

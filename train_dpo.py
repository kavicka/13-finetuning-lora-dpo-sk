"""LoRA + DPO: start from the SFT adapter and teach it which of two answers is better.

Rejected answers are the SFT model's own real mistakes on training reports wherever it makes one, because those
are the errors worth fixing. Where it is already right, a systematic corruption of the gold answer stands in.
The SFT adapter is also the frozen reference, so the KL-like penalty keeps the policy near what already works.
"""

import json
import random
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_tune import DPOConfig, DPOTrainer

from common import gold_json, load_split, messages
from eval_mlx import run
from schema import CLAIM_TYPES, FIELDS, parse_json, score_fields
from train_config import BASE_MODEL, SEED

SFT, OUT = "adapters/sft", "adapters/dpo"
# batch 2: at batch 1 mlx-tune takes a shared-prompt KV-cache path that fails under autograd with this MLX
HP = {
    "beta": 0.1,
    "learning_rate": 1e-5,
    "max_steps": 150,
    "batch_size": 2,
    "max_seq_length": 1024,
    "docs_for_mistakes": 400,
    "seed": SEED,
}


def is_correct(raw: str, rec: dict) -> bool:
    counts = score_fields(parse_json(raw), rec)
    return all(counts[f][1] == 0 and counts[f][2] == 0 for f in FIELDS)


def corrupt(rec: dict, rng: random.Random) -> str:
    """The kinds of wrong answer a small model gives: wrong category, a dropped fact, a flipped yes/no, a lost number."""
    bad = dict(rec)
    kind = rng.choice(["claim_type", "drop_item", "flip", "null_field"])
    if kind == "claim_type":
        bad["claim_type"] = rng.choice([c for c in CLAIM_TYPES if c != rec["claim_type"]])
    elif kind == "drop_item" and len(rec["damaged_items"]) > 1:
        bad["damaged_items"] = rec["damaged_items"][1:]
    elif kind == "flip":
        key = rng.choice(["injured_persons", "police_reported"])
        bad[key] = not rec[key]
    else:
        present = [k for k in ("event_date", "city", "policy_number", "estimated_amount_eur") if rec[k] is not None]
        if present:
            bad[rng.choice(present)] = None
        else:
            bad["injured_persons"] = not rec["injured_persons"]
    return gold_json(bad)


def build_pairs(model, tokenizer, rows: list[dict]) -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    pool = rng.sample(rows, min(HP["docs_for_mistakes"], len(rows)))
    preds = run(model, tokenizer, pool, "train-mistakes")
    pairs, own = [], 0
    for row, pred in zip(pool, preds):
        wrong = not is_correct(pred["raw"], row["record"])
        rejected = pred["raw"].strip() if wrong else corrupt(row["record"], rng)
        own += wrong
        prompt = tokenizer.apply_chat_template(
            messages(row["text"]), add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        pairs.append(
            {"prompt": prompt, "chosen": gold_json(row["record"]) + "<|im_end|>", "rejected": rejected + "<|im_end|>"}
        )
    return pairs, {"pairs": len(pairs), "rejected_is_own_mistake": own, "rejected_is_corruption": len(pairs) - own}


def main() -> None:
    mx.random.seed(SEED)
    model, tokenizer = load(BASE_MODEL, adapter_path=SFT)
    if Path("data/dpo_pairs.json").exists():  # asking the SFT model for its mistakes takes 15 minutes, once is enough
        pairs, info = json.load(open("data/dpo_pairs.json"))
    else:
        pairs, info = build_pairs(model, tokenizer, load_split("train"))
        json.dump([pairs, info], open("data/dpo_pairs.json", "w"), ensure_ascii=False)
    print(info, flush=True)
    model.freeze()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print("trainable LoRA weights", trainable, flush=True)
    args = DPOConfig(beta=HP["beta"], learning_rate=HP["learning_rate"], max_steps=HP["max_steps"], per_device_train_batch_size=HP["batch_size"],
                     gradient_accumulation_steps=1, max_seq_length=HP["max_seq_length"], output_dir="logs/dpo_outputs", logging_steps=10, save_steps=1000)  # fmt: skip
    DPOTrainer(model=model, train_dataset=pairs, tokenizer=tokenizer, args=args).train()
    # saved here rather than by the trainer: this format is what mlx_lm.load and mlx_lm.fuse read back
    Path(OUT).mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{SFT}/adapter_config.json", OUT)
    mx.save_safetensors(f"{OUT}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
    json.dump(
        {"tool": "mlx-tune DPOTrainer", "hyperparameters": HP, "trainable_weights": trainable, **info},
        open("results_train_dpo.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()

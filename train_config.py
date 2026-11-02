"""One place for the training recipe, so results files can log exactly what ran."""

BASE_MODEL = "mlx-community/Qwen3-1.7B-bf16"  # same weights Ollama serves as qwen3:1.7b, before its 4-bit quantisation
SEED = 0
LORA = {
    "model": BASE_MODEL,
    "train": True,
    "data": "data/mlx",
    "adapter_path": "adapters/sft",
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
    "batch_size": 4,
    "iters": 600,  # about 3 passes over 800 reports
    "learning_rate": 1e-4,
    "max_seq_length": 1024,
    "mask_prompt": True,
    "grad_checkpoint": True,
    "steps_per_report": 10,
    "steps_per_eval": 50,
    "val_batches": -1,
    "save_every": 100,
    "seed": SEED,
}

# Tutorial · 13 · Fine-tuning a small model to read Slovak claims: LoRA vs DPO

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial builds the project in the order you would write it: fix the task → make data with known answers → keep the test honest → one prompt and one scorer → baselines → LoRA → DPO → choose on val → ship to Ollama → extra checks. Each step explains **what** the code does, **why** it is written that way, and **how** it works. The README has all the numbers.

**Goal:** a 1.7B model on a Mac that turns a Slovak insurance report into a fixed JSON record, and a fair table comparing zero-shot, few-shot, LoRA and LoRA + DPO.

**You need:** projects 08 (cross-entropy, KL), 09 (low rank and LoRA from scratch) and 12 (how a GPT generates tokens). Tools: Ollama, MLX with `mlx_lm` (LoRA training on the Mac GPU), `mlx-tune` (DPO), llama.cpp (only for the export).

---

## Step 0 · Should this be fine-tuned at all?

| Symptom | Try first |
|---|---|
| The model can do it but needs to be told how | prompt, then a few examples |
| It lacks facts that change or are private | RAG |
| It knows enough, but the **behaviour** is unreliable (format, conventions) | fine-tune |

Every answer here is inside the report, so no outside knowledge is needed. The baselines fail on **behaviour**: the amount as the string `"1950"` instead of a number, a town left in the wrong grammatical case. That is what weights are good at holding, so fine-tuning is the right tool, and the baselines will prove whether it was worth it.

---

## Step 1 · Fix the task first ([`schema.py`](schema.py))

```python
CLAIM_TYPES = ["krupobitie", "víchrica", "voda", "požiar", "krádež", "autonehoda", "iné"]
FIELDS = ["claim_type", "event_date", "city", "policy_number",
          "damaged_items", "estimated_amount_eur", "injured_persons", "police_reported"]
```

**Why first:** everything that decides "correct" lives in one file: the schema, the record sampler, JSON parsing and scoring. All four systems are judged by the same code, so no row of the table can win by a friendlier scorer.

### What "valid JSON" means
```python
def valid_schema(obj):
    if obj is None or set(obj) != set(FIELDS): return False
    return obj["claim_type"] in CLAIM_TYPES and ... isinstance(obj["injured_persons"], bool) ...
```
It must parse **and** every field must exist with the right type (date `YYYY-MM-DD`, amount a number, not a string). `parse_json` forgives a markdown code fence around the JSON, nothing else.

### How fields are scored
```python
def score_fields(pred, gold):      # per field [tp, fp, fn]
    if f == "damaged_items":
        ps, gs = {item_key(i) for i in p}, {item_key(i) for i in g}
        counts[f] = [len(ps & gs), len(ps - gs), len(gs - ps)]
    elif g is None:        counts[f][1] += p is not None    # invented → precision
    elif p is None:        counts[f][2] += 1                # missing → recall
    elif same_scalar(...): counts[f][0] += 1
    else:                  counts[f][1] += 1; counts[f][2] += 1   # wrong → both
```
- **Precision** counts invented and wrong values, **recall** missing and wrong ones; **micro F1** sums tp/fp/fn over all fields of all documents.
- `norm` lowercases and strips diacritics, so "Strecha" = "strecha".
- `item_key` keeps the first 5 letters of each word: Slovak inflects ("strechu" = "strecha"), and a prefix survives that. Amounts must be within 1 %, dates exact.

**Why strict and fixed up front:** a metric tuned after seeing results can make any model look good. The README notes that it is strict on item variants and left it so.

---

## Step 2 · Data with ground truth by construction ([`generate_data.py`](generate_data.py))

```python
rec = sample_record(rng, cities)              # 1. the answer first, from a seeded RNG
text = write(writer, writer_prompt(rec, style, words), seed)   # 2. then a local LLM writes the report
problems = consistency_problems(text, rec)    # 3. then check the text really contains the record
```

**Why the record first:** if you asked an LLM to label existing text, the labels could be wrong. Here the label exists before the text, so it can only be wrong if the writer disobeys, and step 3 catches most of that.

**`sample_record`:** claim type, a date (missing in ~15 %), a city (missing in ~10 %), a policy number like `PZ-123456`, 1–3 items that fit the claim type, an amount from three ranges, and injuries/police with realistic probabilities (police are likely for theft, car accidents and fire).

**`writer_prompt`:** one of six styles (formal e-mail, a hurried note, a form, an angry client, a polite older person, an operator's call record), 40–120 words, "these facts and no other numbers, dates or towns", and an explicit list of what the client does *not* know so it is not invented.

**`consistency_problems`:** every non-null value must appear in the text in some form (policy number, city stem, amount digits, the date as `3. 5.`, `3. mája` or `tretieho mája`, every item, a word for injury, the word police), and null values must stay out (no invented policy number, amount or date).

**Why drawn up front:**
```python
drawn = [(sample_record(rng, cfg["cities"]), rng.choice(STYLES), rng.randint(40, 120)) for _ in range(cfg["n"])]
```
All records are drawn before writing, and each report is appended to the file. A stopped run resumes, and the split never depends on what the writer did.

### Reading by hand
The automatic check passed things that were wrong. Reading every 10th report (105) found two more kinds: a date the writer invented while the label said null, and a date without a year where the label still had one. Both became checks (`date_invented`, `date_no_year`), the data was rebuilt and the baselines rerun. **Automatic checks only find what you thought of; reading finds the rest.**

---

## Step 3 · Keep the test set honest

Three separate defences, all in `SPLITS` and `schema.py`:

1. **A different writer for test:** train and val are written by `qwen3:30b-a3b`, test by `gemma3:12b`. The student cannot score by recognising one writer's style.
2. **Split by seed before any text exists:** train seed 1000, val 2000, test 3000. No report text and no policy number is in both train and test.
3. **Test-only towns:** 10 towns (Trnava, Brezno, Galanta, …) appear only in test. A list of memorised towns is not enough.

`load_split` re-runs the check every time and takes the first 800 / 100 / 150 that pass, so a file never holds a stale verdict.

---

## Step 4 · One prompt and one scorer for everyone ([`common.py`](common.py))

```python
SYSTEM = ("Si systém poisťovne, ktorý z hlásenia poistnej udalosti vyberá údaje. Odpovedz iba jedným JSON objektom ..."
          "city: mesto v 1. páde, alebo null\n" ...)
def gold_json(rec):  return json.dumps({f: rec[f] for f in FIELDS}, ensure_ascii=False)
```

- **The same system prompt** in zero-shot, few-shot, training and evaluation. The only difference between rows is what the model has learned or been shown.
- **`gold_json`** always uses the same key order, so the student does not spend weights on formatting noise.
- **`few_shots`:** 3 training reports chosen by a fixed seed (7) *before* any result existed, at most one per claim type for variety.
- **`summarise`:** micro precision/recall/F1, per-field F1, % valid JSON, median latency, and a **bootstrap**: 1,000 times, draw 150 documents with replacement and recompute F1; the 2.5th and 97.5th percentiles give the 95 % interval. **Why:** with 150 documents, 0.964 vs 0.968 could be luck. The interval says how much F1 would move with a different sample of documents.

---

## Step 5 · Baselines in Ollama ([`eval_ollama.py`](eval_ollama.py))

```python
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 300}
body = {"model": model, "messages": msgs, "stream": False, "think": False, ...}
chat(model, messages(test[0]["text"], shots))   # warm-up: load the model before timing
```

- **Temperature 0:** greedy, so the same input always gives the same answer.
- **`think: False`:** Qwen3 can "think" before answering; that would cost time and break the JSON-only output.
- **Warm-up:** the first call loads the model into memory, which should not be charged to document 1.

**Result:** zero-shot **0.812** F1 but only **24.0 %** valid JSON; few-shot (3 examples) **0.885**, 87.3 % valid. The model can mostly read the report; it just does not follow the format reliably.

---

## Step 6 · LoRA supervised fine-tuning ([`train_sft.py`](train_sft.py), [`train_config.py`](train_config.py))

```python
LORA = {"model": "mlx-community/Qwen3-1.7B-bf16", "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
        "batch_size": 4, "iters": 600, "learning_rate": 1e-4, "mask_prompt": True, "grad_checkpoint": True, ...}
```

**What LoRA does** (project 09): the original weights W stay frozen; each adapted layer learns a small update B·A of rank 8, added as W + scale·B·A. Here the last 16 layers: **4.98M trainable weights out of 1,720M (0.289 %)**. Training fits in memory, and the adapter is a small file.

**The data format:**
```python
msgs = messages(r["text"]) + [{"role": "assistant", "content": gold_json(r["record"])}]
```
Each example is a full chat: system prompt, the report as the user message, the gold JSON as the answer. The same `messages()` as at evaluation time, so training and use look identical.

**`mask_prompt: True`:** the loss is counted only on the answer tokens. **Why:** the report is context, not something the model should learn to reproduce; without masking most of the gradient would go into predicting the report text.

**Other choices:**
- `grad_checkpoint`: recompute activations in the backward pass instead of storing them, less memory for a bit more time.
- `iters 600 × batch 4` ≈ 3 passes over 800 reports.
- One config dict in one file, dumped to `logs/lora.yaml` and to `results_train_sft.json`, so the results always record exactly what ran.

**How it runs:** `train_sft.py` writes `data/mlx/train.jsonl` and `valid.jsonl`, then starts `python -m mlx_lm lora --config …` as a subprocess, streams its output and parses the loss table with a regex (after stripping colour codes) into the loss curve. Validation loss falls from 0.930 to 0.020 in 50 steps and ends at 0.011.

**Result:** **0.968** F1 (0.959–0.977), 99.3 % valid JSON.

---

## Step 7 · DPO on top of SFT ([`train_dpo.py`](train_dpo.py))

**The idea:** supervised training says "this answer is right". **DPO** (Direct Preference Optimisation) says "this answer is better than that one": it raises the probability of the *chosen* answer relative to the *rejected* one, measured against a frozen **reference** model (here the SFT adapter). `beta` (0.1) controls how far the policy may move from the reference, a KL-like leash (project 08).

### Building the pairs
```python
preds = run(model, tokenizer, pool, "train-mistakes")       # the SFT model on 400 training reports
wrong = not is_correct(pred["raw"], row["record"])
rejected = pred["raw"].strip() if wrong else corrupt(row["record"], rng)
pairs.append({"prompt": prompt, "chosen": gold_json(rec) + "<|im_end|>", "rejected": rejected + "<|im_end|>"})
```
- **Plan:** use the SFT model's *own* mistakes as rejected answers, because those are the errors worth fixing.
- **Reality:** it made only **11** mistakes on 400 reports, so 389 pairs use `corrupt`: a wrong claim type, a dropped item, a flipped yes/no, a field set to null.
- `<|im_end|>` is Qwen's end-of-turn token: the model should also learn where to stop.
- The pairs are saved to `data/dpo_pairs.json` because asking for the mistakes takes 15 minutes.

### Training only the LoRA weights
```python
model.freeze()
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.unfreeze(keys=["lora_a", "lora_b"], strict=False)
DPOTrainer(model=model, train_dataset=pairs, tokenizer=tokenizer, args=DPOConfig(beta=0.1, learning_rate=1e-5, max_steps=150, ...)).train()
mx.save_safetensors(f"{OUT}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
```
Load the base model with the SFT adapter, freeze everything, unfreeze only the LoRA matrices A and B, train. The adapter is saved by hand in the format `mlx_lm.load` and `mlx_lm.fuse` read. Batch 2, because batch 1 crashed in mlx-tune's shared-prompt cache path with this MLX version. The learning rate is 10× lower than SFT: DPO only nudges a model that already works.

**Result:** 0.964 vs 0.968, inside the noise. **Why nothing:** a model that is nearly right has nothing to learn from synthetic mistakes it would never make. DPO fits subjective qualities (tone, helpfulness) better than a task with one correct answer.

---

## Step 8 · Choose on val, never on test ([`eval_mlx.py`](eval_mlx.py))

```bash
uv run python eval_mlx.py val_lora     --adapter adapters/sft --split val
uv run python eval_mlx.py val_lora_dpo --adapter adapters/dpo --split val
```

**The rule, set in advance:** take the higher F1 if the 95 % intervals do not overlap; otherwise take the simpler model. On val: SFT 0.982, DPO 0.984, intervals overlap → **plain LoRA**. The choice is written in `results_best.json`. Test is used only to report, so the reported number is not optimistic.

`eval_mlx.py` uses `apply_chat_template(..., enable_thinking=False)`, the same messages and the same `save_eval` as Ollama, so the MLX rows are scored identically.

---

## Step 9 · Ship it to Ollama ([`fuse.sh`](fuse.sh), [`Modelfile`](Modelfile))

```sh
python -m mlx_lm fuse --model mlx-community/Qwen3-1.7B-bf16 --adapter-path adapters/sft --save-path models/fused
python "$LLAMA_CPP/convert_hf_to_gguf.py" models/fused --outfile models/fused-f16.gguf --outtype f16
ollama create claims-sk -q q4_K_M -f Modelfile
```

1. **Fuse:** compute W + scale·B·A once and save plain weights, so no adapter is needed at runtime.
2. **Convert to GGUF with llama.cpp:** Ollama rejects Qwen3 safetensors and `mlx_lm` has no GGUF export for Qwen3, so llama.cpp's converter sits in between.
3. **Quantise to Q4_K_M:** the same 4-bit level as `qwen3:1.7b`, so the latency rows are comparable. The `Modelfile` carries Qwen3's chat template and stop tokens; the evaluation passes temperature 0 itself.

**Result:** `claims-sk` (1.1 GB) reaches **0.966** F1, **100 %** valid JSON and **0.83 s** per report, faster than zero-shot (1.02 s) at the same quantisation. Its answers are single-line and shorter. 4-bit cost almost nothing (0.968 → 0.966).

---

## Step 10 · Extra checks

### How much data? ([`learning_curve.py`](learning_curve.py))
The same recipe on the first 25, 100 and 400 reports, 300 iterations each (same steps, so a small set is seen many more times): 0.916, 0.952, 0.970. **25 examples already beat the few-shot prompt.** Measure how little will do before collecting more.

### Did it forget everything else? ([`forgetting.py`](forgetting.py))
Eight plain Slovak questions, no system prompt, base vs fine-tuned. The check looks mainly for `as_json`: a tuned model that answers everything with JSON. Neither did. The keyword score is crude (it counted a wrong leap-year answer as right), so read the outputs instead of trusting the score.

### Where are the errors? ([`analysis.py`](analysis.py))
`damaged_items` and `city` are the weakest fields. Cities: 97 of 99 right on towns seen in training, 22 of 36 on test-only towns, mostly a wrong grammatical case ("Brezne" for Brezno). Part of the gain is memorised names, which the test-only towns made visible.

---

## Step 11 · Run everything, resumably ([`pipeline.sh`](pipeline.sh))

```sh
[ -f results_eval_zero_shot.json ] || uv run python -u eval_ollama.py zero_shot > logs/eval_zero_shot.log 2>&1
```
Every stage skips itself when its result file exists, and data generation appends as it goes. After a sleeping Mac or a dead terminal, run `nohup caffeinate -i sh pipeline.sh …` again and it continues.

---

## Try it yourself

1. Train LoRA with `"mask_prompt": False`. Does F1 change, and what happens to the training loss values?
2. Change `rank` to 2 and 32 in `train_config.py`. With 800 examples, does a larger rank help, or is rank 8 already enough?
3. Make DPO pairs only from real mistakes: run the SFT model on all 800 training reports and keep only wrong answers. How many pairs do you get, and is that enough to train on?

## Recap

| Step | Idea |
|---|---|
| 0 | Fine-tune for behaviour; prompt first; RAG for facts |
| 1 | One file decides "correct": schema, parsing, strict scoring fixed in advance |
| 2 | Label first, then text; automatic check plus reading by hand |
| 3 | Different writer, split by seed, test-only towns |
| 4 | Same prompt and scorer for every row; bootstrap interval |
| 5 | Baselines fail on format (24 % valid JSON zero-shot) |
| 6 | LoRA rank 8, 0.289 % of weights, loss only on the answer → 0.968 |
| 7 | DPO = chosen vs rejected against a reference; no real mistakes → no gain |
| 8 | Choose on val with a rule set in advance; test only reports |
| 9 | Fuse → GGUF → Q4 in Ollama: 0.966, 100 % valid, 0.83 s |

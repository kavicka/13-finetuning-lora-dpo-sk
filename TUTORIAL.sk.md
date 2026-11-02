# Tutoriál · 13 · Fine-tuning malého modelu na čítanie slovenských hlásení: LoRA vs DPO

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt v poradí, v akom by si ho písal: zafixuj úlohu → vyrob dáta so známymi odpoveďami → udrž test poctivý → jeden prompt a jeden hodnotiaci kód → baseliny → LoRA → DPO → výber na validácii → nasadenie do Ollamy → doplnkové kontroly. Každý krok vysvetľuje, **čo** kód robí, **prečo** je napísaný takto a **ako** funguje. README má všetky čísla.

**Cieľ:** 1,7B model na Macu, ktorý zmení slovenské hlásenie poistnej udalosti na pevný JSON záznam, a férová tabuľka porovnávajúca zero-shot, few-shot, LoRA a LoRA + DPO.

**Potrebuješ:** projekty 08 (cross-entropy, KL), 09 (nízka hodnosť a LoRA od nuly) a 12 (ako GPT generuje tokeny). Nástroje: Ollama, MLX s `mlx_lm` (tréning LoRA na GPU Macu), `mlx-tune` (DPO), llama.cpp (len na export).

---

## Krok 0 · Treba to vôbec fine-tunovať?

| Symptóm | Skús najprv |
|---|---|
| Model to vie, ale treba mu povedať ako | prompt, potom pár príkladov |
| Chýbajú mu fakty, ktoré sa menia alebo sú súkromné | RAG |
| Vie dosť, ale **správanie** je nespoľahlivé (formát, konvencie) | fine-tuning |

Každá odpoveď je priamo v hlásení, takže netreba žiadne vonkajšie znalosti. Baseliny zlyhávajú na **správaní**: suma ako reťazec `"1950"` namiesto čísla, mesto ponechané v nesprávnom páde. Práve správanie si váhy dobre pamätajú, takže fine-tuning je správny nástroj a baseliny ukážu, či sa oplatil.

---

## Krok 1 · Najprv zafixuj úlohu ([`schema.py`](schema.py))

```python
CLAIM_TYPES = ["krupobitie", "víchrica", "voda", "požiar", "krádež", "autonehoda", "iné"]
FIELDS = ["claim_type", "event_date", "city", "policy_number",
          "damaged_items", "estimated_amount_eur", "injured_persons", "police_reported"]
```

**Prečo najprv:** všetko, čo rozhoduje o „správne“, je v jednom súbore: schéma, generátor záznamov, parsovanie JSON a hodnotenie. Všetky štyri systémy hodnotí ten istý kód, takže žiadny riadok tabuľky nemôže vyhrať vďaka miernejšiemu hodnoteniu.

### Čo znamená „platný JSON“
```python
def valid_schema(obj):
    if obj is None or set(obj) != set(FIELDS): return False
    return obj["claim_type"] in CLAIM_TYPES and ... isinstance(obj["injured_persons"], bool) ...
```
Musí sa dať sparsovať **a** každé pole musí existovať so správnym typom (dátum `YYYY-MM-DD`, suma číslo, nie reťazec). `parse_json` odpustí markdown blok kódu okolo JSON, nič iné.

### Ako sa hodnotia polia
```python
def score_fields(pred, gold):      # pre každé pole [tp, fp, fn]
    if f == "damaged_items":
        ps, gs = {item_key(i) for i in p}, {item_key(i) for i in g}
        counts[f] = [len(ps & gs), len(ps - gs), len(gs - ps)]
    elif g is None:        counts[f][1] += p is not None    # vymyslené → precision
    elif p is None:        counts[f][2] += 1                # chýba → recall
    elif same_scalar(...): counts[f][0] += 1
    else:                  counts[f][1] += 1; counts[f][2] += 1   # nesprávne → oboje
```
- **Precision** počíta vymyslené a nesprávne hodnoty, **recall** chýbajúce a nesprávne; **micro F1** sčíta tp/fp/fn cez všetky polia všetkých dokumentov.
- `norm` zmení na malé písmená a odstráni diakritiku, takže „Strecha“ = „strecha“.
- `item_key` ponechá prvých 5 písmen každého slova: slovenčina skloňuje („strechu“ = „strecha“) a predpona to prežije. Sumy musia sedieť na 1 %, dátumy presne.

**Prečo prísne a vopred:** metrika vyladená po pohľade na výsledky dokáže, aby vyzeral dobre ktorýkoľvek model. README spomína, že pri variantoch vecí je prísna a zostala tak.

---

## Krok 2 · Dáta so správnou odpoveďou z konštrukcie ([`generate_data.py`](generate_data.py))

```python
rec = sample_record(rng, cities)              # 1. najprv odpoveď, zo seedovaného generátora
text = write(writer, writer_prompt(rec, style, words), seed)   # 2. potom lokálny LLM napíše hlásenie
problems = consistency_problems(text, rec)    # 3. potom over, že text naozaj obsahuje záznam
```

**Prečo najprv záznam:** keby si LLM nechal označiť existujúci text, označenia by mohli byť zlé. Tu označenie existuje pred textom, takže zlé môže byť len vtedy, keď pisateľ neposlúchne, a krok 3 to väčšinou zachytí.

**`sample_record`:** typ udalosti, dátum (chýba v ~15 %), mesto (chýba v ~10 %), číslo poistky ako `PZ-123456`, 1–3 veci, ktoré k typu sedia, suma z troch rozsahov a zranenia/polícia s realistickými pravdepodobnosťami (polícia je pravdepodobná pri krádeži, autonehode a požiari).

**`writer_prompt`:** jeden zo šiestich štýlov (formálny e-mail, narýchlo napísaná správa, formulár, rozčúlený klient, zdvorilý starší človek, záznam operátora), 40–120 slov, „tieto fakty a žiadne iné čísla, dátumy ani mestá“ a výslovný zoznam toho, čo klient *nepozná*, aby si to nevymýšľal.

**`consistency_problems`:** každá hodnota, ktorá nie je null, sa musí v texte objaviť v nejakej podobe (číslo poistky, kmeň mesta, číslice sumy, dátum ako `3. 5.`, `3. mája` alebo `tretieho mája`, každá vec, slovo pre zranenie, slovo polícia) a hodnoty null musia z textu zostať vonku (žiadne vymyslené číslo poistky, suma ani dátum).

**Prečo vylosované vopred:**
```python
drawn = [(sample_record(rng, cfg["cities"]), rng.choice(STYLES), rng.randint(40, 120)) for _ in range(cfg["n"])]
```
Všetky záznamy sa vylosujú pred písaním a každé hlásenie sa hneď pripíše do súboru. Zastavený beh pokračuje a rozdelenie nikdy nezávisí od toho, čo spravil pisateľ.

### Čítanie ručne
Automatická kontrola prepustila veci, ktoré boli zlé. Prečítanie každého 10. hlásenia (105) našlo ďalšie dva druhy: dátum, ktorý pisateľ vymyslel, hoci záznam mal null, a dátum bez roku, kde záznam rok stále mal. Z oboch sa stali kontroly (`date_invented`, `date_no_year`), dáta sa prestavali a baseliny sa spustili znova. **Automatické kontroly nájdu len to, na čo si pomyslel; čítanie nájde zvyšok.**

---

## Krok 3 · Udrž testovaciu sadu poctivú

Tri samostatné obrany, všetky v `SPLITS` a `schema.py`:

1. **Iný pisateľ pre test:** train a val píše `qwen3:30b-a3b`, test `gemma3:12b`. Študent nemôže skórovať tým, že spozná štýl jedného pisateľa.
2. **Rozdelenie podľa seedu skôr, než existuje text:** train seed 1000, val 2000, test 3000. Žiadny text hlásenia a žiadne číslo poistky nie je v train aj v teste.
3. **Mestá len v teste:** 10 miest (Trnava, Brezno, Galanta, …) je len v teste. Zoznam zapamätaných miest nestačí.

`load_split` zakaždým znova spustí kontrolu a vezme prvých 800 / 100 / 150, ktoré prejdú, takže súbor nikdy nedrží zastaraný verdikt.

---

## Krok 4 · Jeden prompt a jedno hodnotenie pre všetkých ([`common.py`](common.py))

```python
SYSTEM = ("Si systém poisťovne, ktorý z hlásenia poistnej udalosti vyberá údaje. Odpovedz iba jedným JSON objektom ..."
          "city: mesto v 1. páde, alebo null\n" ...)
def gold_json(rec):  return json.dumps({f: rec[f] for f in FIELDS}, ensure_ascii=False)
```

- **Ten istý systémový prompt** pri zero-shot, few-shot, tréningu aj vyhodnotení. Jediný rozdiel medzi riadkami je to, čo sa model naučil alebo čo mu bolo ukázané.
- **`gold_json`** používa vždy rovnaké poradie kľúčov, aby študent nemíňal váhy na šum vo formátovaní.
- **`few_shots`:** 3 trénovacie hlásenia vybrané pevným seedom (7) *skôr*, než existoval akýkoľvek výsledok, najviac jedno na typ udalosti kvôli rozmanitosti.
- **`summarise`:** micro precision/recall/F1, F1 pre každé pole, % platného JSON, medián latencie a **bootstrap**: 1 000-krát vyber 150 dokumentov s opakovaním a prepočítaj F1; 2,5. a 97,5. percentil dajú 95 % interval. **Prečo:** pri 150 dokumentoch môže byť 0,964 vs 0,968 náhoda. Interval hovorí, o koľko by sa F1 pohlo pri inej vzorke dokumentov.

---

## Krok 5 · Baseliny v Ollame ([`eval_ollama.py`](eval_ollama.py))

```python
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 300}
body = {"model": model, "messages": msgs, "stream": False, "think": False, ...}
chat(model, messages(test[0]["text"], shots))   # zahriatie: načítaj model pred meraním
```

- **Teplota 0:** greedy, takže rovnaký vstup dá vždy rovnakú odpoveď.
- **`think: False`:** Qwen3 vie pred odpoveďou „premýšľať“; stálo by to čas a pokazilo výstup, ktorý má byť len JSON.
- **Zahriatie:** prvé volanie načíta model do pamäte, a to sa nemá pripísať dokumentu 1.

**Výsledok:** zero-shot **0,812** F1, ale len **24,0 %** platného JSON; few-shot (3 príklady) **0,885**, 87,3 % platných. Model hlásenie väčšinou prečíta; len spoľahlivo nedodrží formát.

---

## Krok 6 · LoRA, učenie s učiteľom ([`train_sft.py`](train_sft.py), [`train_config.py`](train_config.py))

```python
LORA = {"model": "mlx-community/Qwen3-1.7B-bf16", "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
        "batch_size": 4, "iters": 600, "learning_rate": 1e-4, "mask_prompt": True, "grad_checkpoint": True, ...}
```

**Čo robí LoRA** (projekt 09): pôvodné váhy W zostanú zmrazené; každá upravená vrstva sa naučí malú zmenu B·A s hodnosťou 8, pripočítanú ako W + scale·B·A. Tu v posledných 16 vrstvách: **4,98M trénovateľných váh z 1 720M (0,289 %)**. Tréning sa zmestí do pamäte a adaptér je malý súbor.

**Formát dát:**
```python
msgs = messages(r["text"]) + [{"role": "assistant", "content": gold_json(r["record"])}]
```
Každý príklad je celý chat: systémový prompt, hlásenie ako správa používateľa, správny JSON ako odpoveď. Tá istá funkcia `messages()` ako pri vyhodnotení, takže tréning a použitie vyzerajú rovnako.

**`mask_prompt: True`:** loss sa počíta len na tokenoch odpovede. **Prečo:** hlásenie je kontext, nie niečo, čo sa má model naučiť reprodukovať; bez maskovania by väčšina gradientu išla do predpovedania textu hlásenia.

**Ďalšie voľby:**
- `grad_checkpoint`: prepočítaj aktivácie pri spätnom prechode namiesto ich ukladania, menej pamäte za trochu viac času.
- `iters 600 × batch 4` ≈ 3 prechody cez 800 hlásení.
- Jedna konfigurácia v jednom súbore, uložená do `logs/lora.yaml` aj do `results_train_sft.json`, takže výsledky vždy zaznamenajú presne to, čo bežalo.

**Ako to beží:** `train_sft.py` zapíše `data/mlx/train.jsonl` a `valid.jsonl`, potom spustí `python -m mlx_lm lora --config …` ako podproces, streamuje jeho výstup a regexom (po odstránení farebných kódov) vytiahne tabuľku loss do krivky. Validačná loss klesne z 0,930 na 0,020 za 50 krokov a skončí na 0,011.

**Výsledok:** **0,968** F1 (0,959–0,977), 99,3 % platného JSON.

---

## Krok 7 · DPO nad SFT ([`train_dpo.py`](train_dpo.py))

**Myšlienka:** učenie s učiteľom hovorí „táto odpoveď je správna“. **DPO** (Direct Preference Optimisation) hovorí „táto odpoveď je lepšia ako tamtá“: zvýši pravdepodobnosť *vybranej* odpovede voči *odmietnutej*, meranú oproti zmrazenému **referenčnému** modelu (tu SFT adaptér). `beta` (0,1) určuje, ako ďaleko sa smie model od referencie vzdialiť, vôdzka podobná KL (projekt 08).

### Tvorba dvojíc
```python
preds = run(model, tokenizer, pool, "train-mistakes")       # SFT model na 400 trénovacích hláseniach
wrong = not is_correct(pred["raw"], row["record"])
rejected = pred["raw"].strip() if wrong else corrupt(row["record"], rng)
pairs.append({"prompt": prompt, "chosen": gold_json(rec) + "<|im_end|>", "rejected": rejected + "<|im_end|>"})
```
- **Plán:** ako odmietnuté odpovede použiť *vlastné* chyby SFT modelu, lebo práve tie sa oplatí opraviť.
- **Realita:** na 400 hláseniach spravil len **11** chýb, takže 389 dvojíc používa `corrupt`: nesprávny typ udalosti, vynechaná vec, otočené áno/nie, pole nastavené na null.
- `<|im_end|>` je Qwenov token konca odpovede: model sa má naučiť aj to, kde skončiť.
- Dvojice sa uložia do `data/dpo_pairs.json`, lebo zbieranie chýb trvá 15 minút.

### Tréning len LoRA váh
```python
model.freeze()
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.unfreeze(keys=["lora_a", "lora_b"], strict=False)
DPOTrainer(model=model, train_dataset=pairs, tokenizer=tokenizer, args=DPOConfig(beta=0.1, learning_rate=1e-5, max_steps=150, ...)).train()
mx.save_safetensors(f"{OUT}/adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))
```
Načítaj základný model so SFT adaptérom, všetko zmraz, rozmraz len LoRA matice A a B, trénuj. Adaptér sa uloží ručne vo formáte, ktorý čítajú `mlx_lm.load` a `mlx_lm.fuse`. Batch 2, lebo pri batchi 1 mlx-tune spadol v ceste so zdieľanou cache promptu s touto verziou MLX. Learning rate je 10× nižší ako pri SFT: DPO len postrčí model, ktorý už funguje.

**Výsledok:** 0,964 vs 0,968, v rámci šumu. **Prečo nič:** model, ktorý je takmer správny, sa nemá čo naučiť zo syntetických chýb, ktoré by nikdy nespravil. DPO sa hodí skôr na subjektívne vlastnosti (tón, užitočnosť) než na úlohu s jednou správnou odpoveďou.

---

## Krok 8 · Vyberaj na validácii, nikdy na teste ([`eval_mlx.py`](eval_mlx.py))

```bash
uv run python eval_mlx.py val_lora     --adapter adapters/sft --split val
uv run python eval_mlx.py val_lora_dpo --adapter adapters/dpo --split val
```

**Pravidlo, stanovené vopred:** vezmi vyššie F1, ak sa 95 % intervaly neprekrývajú; inak vezmi jednoduchší model. Na validácii: SFT 0,982, DPO 0,984, intervaly sa prekrývajú → **obyčajná LoRA**. Voľba je zapísaná v `results_best.json`. Test slúži len na reportovanie, takže reportované číslo nie je optimistické.

`eval_mlx.py` používa `apply_chat_template(..., enable_thinking=False)`, tie isté správy a tú istú `save_eval` ako Ollama, takže MLX riadky sa hodnotia rovnako.

---

## Krok 9 · Nasaď do Ollamy ([`fuse.sh`](fuse.sh), [`Modelfile`](Modelfile))

```sh
python -m mlx_lm fuse --model mlx-community/Qwen3-1.7B-bf16 --adapter-path adapters/sft --save-path models/fused
python "$LLAMA_CPP/convert_hf_to_gguf.py" models/fused --outfile models/fused-f16.gguf --outtype f16
ollama create claims-sk -q q4_K_M -f Modelfile
```

1. **Zlúčenie (fuse):** raz vypočítaj W + scale·B·A a ulož obyčajné váhy, takže pri behu netreba adaptér.
2. **Konverzia do GGUF cez llama.cpp:** Ollama odmietne Qwen3 safetensors a `mlx_lm` nemá pre Qwen3 export do GGUF, preto je medzi nimi konvertor z llama.cpp.
3. **Kvantizácia na Q4_K_M:** rovnaká 4-bitová úroveň ako `qwen3:1.7b`, takže riadky s latenciou sa dajú porovnať. `Modelfile` nesie chat šablónu a stop tokeny Qwen3; vyhodnotenie si teplotu 0 posiela samo.

**Výsledok:** `claims-sk` (1,1 GB) dosiahne **0,966** F1, **100 %** platného JSON a **0,83 s** na hlásenie, rýchlejšie ako zero-shot (1,02 s) pri rovnakej kvantizácii. Jeho odpovede sú na jednom riadku a kratšie. 4 bity nestáli takmer nič (0,968 → 0,966).

---

## Krok 10 · Doplnkové kontroly

### Koľko dát treba? ([`learning_curve.py`](learning_curve.py))
Ten istý recept na prvých 25, 100 a 400 hláseniach, každý 300 iterácií (rovnaký počet krokov, takže malú sadu model uvidí oveľa viackrát): 0,916, 0,952, 0,970. **Už 25 príkladov porazí few-shot prompt.** Zmeraj, koľko málo stačí, skôr než budeš zbierať viac.

### Zabudol všetko ostatné? ([`forgetting.py`](forgetting.py))
Osem obyčajných slovenských otázok, bez systémového promptu, základný vs vyladený model. Kontrola hľadá hlavne `as_json`: vyladený model, ktorý na všetko odpovie JSON-om. Ani jeden to nespravil. Skóre podľa kľúčových slov je hrubé (nesprávnu odpoveď o priestupnom roku započítalo ako správnu), takže čítaj výstupy namiesto dôvery v skóre.

### Kde sú chyby? ([`analysis.py`](analysis.py))
Najslabšie polia sú `damaged_items` a `city`. Mestá: 97 z 99 správne pri mestách z tréningu, 22 z 36 pri mestách len z testu, väčšinou nesprávny pád („Brezne“ namiesto Brezno). Časť zlepšenia sú zapamätané názvy, čo ukázali práve mestá len z testu.

---

## Krok 11 · Spusti všetko, s pokračovaním ([`pipeline.sh`](pipeline.sh))

```sh
[ -f results_eval_zero_shot.json ] || uv run python -u eval_ollama.py zero_shot > logs/eval_zero_shot.log 2>&1
```
Každá fáza sa preskočí, keď jej výsledkový súbor existuje, a generovanie dát pripisuje priebežne. Po uspatom Macu alebo zatvorenom termináli spusti `nohup caffeinate -i sh pipeline.sh …` znova a pokračuje.

---

## Vyskúšaj sám

1. Natrénuj LoRA s `"mask_prompt": False`. Zmení sa F1 a čo sa stane s hodnotami trénovacej loss?
2. Zmeň `rank` na 2 a 32 v `train_config.py`. Pomôže pri 800 príkladoch väčšia hodnosť, alebo stačí 8?
3. Sprav DPO dvojice len zo skutočných chýb: pusti SFT model na všetkých 800 trénovacích hláseniach a ponechaj len nesprávne odpovede. Koľko dvojíc dostaneš a stačí to na tréning?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 0 | Fine-tuning na správanie; najprv prompt; RAG na fakty |
| 1 | O „správne“ rozhoduje jeden súbor: schéma, parsovanie, prísne hodnotenie vopred |
| 2 | Najprv označenie, potom text; automatická kontrola plus čítanie ručne |
| 3 | Iný pisateľ, rozdelenie podľa seedu, mestá len v teste |
| 4 | Rovnaký prompt a hodnotenie pre každý riadok; bootstrap interval |
| 5 | Baseliny zlyhávajú na formáte (24 % platného JSON pri zero-shot) |
| 6 | LoRA hodnosti 8, 0,289 % váh, loss len na odpovedi → 0,968 |
| 7 | DPO = vybraná vs odmietnutá voči referencii; bez skutočných chýb → bez zisku |
| 8 | Vyberaj na validácii podľa pravidla stanoveného vopred; test len reportuje |
| 9 | Zlúčenie → GGUF → Q4 v Ollame: 0,966, 100 % platných, 0,83 s |

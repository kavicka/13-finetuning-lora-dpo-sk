"""The extraction task: schema, the seeded record sampler, parsing and field-level scoring.

Everything that decides what counts as correct lives here, so all four systems are judged by the same code.
"""

import json
import random
import re
import unicodedata

CLAIM_TYPES = ["krupobitie", "víchrica", "voda", "požiar", "krádež", "autonehoda", "iné"]
FIELDS = [
    "claim_type",
    "event_date",
    "city",
    "policy_number",
    "damaged_items",
    "estimated_amount_eur",
    "injured_persons",
    "police_reported",
]

TRAIN_CITIES = ["Bratislava", "Košice", "Prešov", "Žilina", "Nitra", "Martin", "Trenčín", "Poprad", "Prievidza", "Zvolen", "Považská Bystrica", "Michalovce", "Levice", "Komárno", "Humenné", "Bardejov", "Liptovský Mikuláš", "Ružomberok", "Piešťany", "Topoľčany", "Čadca", "Dunajská Streda", "Pezinok", "Trebišov", "Lučenec", "Rimavská Sobota", "Dubnica nad Váhom", "Vranov nad Topľou", "Partizánske", "Šaľa"]  # fmt: skip
# cities only the test set uses: the student cannot pass by memorising a list of towns
TEST_ONLY_CITIES = [
    "Trnava",
    "Banská Bystrica",
    "Senica",
    "Skalica",
    "Kežmarok",
    "Snina",
    "Brezno",
    "Detva",
    "Galanta",
    "Malacky",
]

ITEMS = {
    "krupobitie": ["strecha", "auto", "okná", "skleník", "fasáda", "solárne panely", "markíza", "odkvapy", "záhradný nábytok"],
    "víchrica": ["strecha", "plot", "komín", "strom", "garáž", "anténa", "pergola", "bazén", "antuka"],
    "voda": ["podlaha", "kuchynská linka", "koberec", "práčka", "omietka", "nábytok", "dvere", "kotol", "knihy"],
    "požiar": ["kuchyňa", "strecha", "garáž", "nábytok", "elektroinštalácia", "televízor", "sklad", "záclony", "drevený altánok"],
    "krádež": ["bicykel", "notebook", "šperky", "hotovosť", "kosačka", "mobil", "náradie", "kamera", "kolobežka"],
    "autonehoda": ["nárazník", "dvere", "svetlomet", "kapota", "čelné sklo", "blatník", "zrkadlo", "disk", "chladič"],
    "iné": ["výklad", "potrubie", "bránka", "schody", "lampa", "tabuľa", "klimatizácia", "žalúzie", "dron"],
}  # fmt: skip

STYLES = [
    "formálny e-mail poisťovni",
    "krátka stručná správa napísaná v rýchlosti, bez oslovenia",
    "vyplnený formulár s riadkami typu popis udalosti, miesto, poistka, škoda",
    "rozčúlený klient, ktorý sa sťažuje a hovorí v dlhších vetách",
    "starší človek, ktorý píše zdvorilo a trochu zoširoka",
    "záznam operátora z telefonátu v tretej osobe",
]


_UNITS = ["prveho", "druheho", "tretieho", "stvrteho", "piateho", "siesteho", "siedmeho", "osmeho", "deviateho", "desiateho", "jedenasteho", "dvanasteho", "trinasteho", "strnasteho", "patnasteho", "sestnasteho", "sedemnasteho", "osemnasteho", "devatnasteho", "dvadsiateho"]  # fmt: skip
# days as a writer may spell them out: "osemnásteho februára", day 1 to 28
ORDINALS = {i + 1: w for i, w in enumerate(_UNITS)} | {20 + i: f"dvadsiateho\\s*{_UNITS[i - 1]}" for i in range(1, 9)}


def sample_record(rng: random.Random, cities: list[str]) -> dict:
    claim_type = rng.choice(CLAIM_TYPES)
    has_date, has_amount = rng.random() > 0.15, rng.random() > 0.2
    year, month, day = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
    prefix = rng.choice(["PZ", "HD", "AUT", "MAJ"])
    amount = rng.choice([rng.randrange(150, 2000, 10), rng.randrange(2000, 30000, 100), rng.randrange(300, 9000, 50)])
    police_likely = claim_type in ("krádež", "autonehoda", "požiar")
    return {
        "claim_type": claim_type,
        "event_date": f"{year}-{month:02d}-{day:02d}" if has_date else None,
        "city": rng.choice(cities) if rng.random() > 0.1 else None,
        "policy_number": f"{prefix}-{rng.randint(100000, 999999)}" if rng.random() > 0.2 else None,
        "damaged_items": rng.sample(ITEMS[claim_type], rng.randint(1, 3)),
        "estimated_amount_eur": amount if has_amount else None,
        "injured_persons": rng.random() < (0.3 if claim_type in ("autonehoda", "požiar") else 0.05),
        "police_reported": rng.random() < (0.7 if police_likely else 0.1),
    }


def norm(text: str) -> str:
    """Lowercase without diacritics: "Strecha" and "strecha", "kosačka" and "kosacka" are one answer."""
    text = unicodedata.normalize("NFKD", str(text).strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def stem(text: str, n: int = 4) -> str:
    """Slovak inflects, so a report says "v Trnave" or "strechu". A prefix survives that."""
    return norm(text)[:n]


def parse_json(raw: str) -> dict | None:
    """The first JSON object in the output. Code fences are forgiven; anything else is a failure."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[raw.index("{") :])
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_schema(obj: dict | None) -> bool:
    """% valid JSON means this: it parses AND every field exists with the right type."""
    if obj is None or set(obj) != set(FIELDS):
        return False
    date = obj["event_date"]
    amount = obj["estimated_amount_eur"]
    items = obj["damaged_items"]
    return (
        obj["claim_type"] in CLAIM_TYPES
        and (date is None or (isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) is not None))
        and (obj["city"] is None or isinstance(obj["city"], str))
        and (obj["policy_number"] is None or isinstance(obj["policy_number"], str))
        and isinstance(items, list)
        and all(isinstance(i, str) for i in items)
        and (amount is None or (isinstance(amount, (int, float)) and not isinstance(amount, bool)))
        and isinstance(obj["injured_persons"], bool)
        and isinstance(obj["police_reported"], bool)
    )


def same_scalar(field: str, pred, gold) -> bool:
    if field == "estimated_amount_eur":
        if isinstance(pred, str):
            digits = re.sub(r"[^\d.]", "", pred.replace(",", "."))
            pred = float(digits) if re.fullmatch(r"\d+(\.\d+)?", digits) else None
        if pred is None or isinstance(pred, bool) or not isinstance(pred, (int, float)):
            return False
        return abs(pred - gold) <= 0.01 * abs(gold)
    if field in ("city", "policy_number", "claim_type"):
        return norm(pred) == norm(gold)
    return pred == gold  # dates exact, booleans exact


def item_key(item) -> str:
    """Set comparison of nouns tolerates inflection and nothing more: "strechu" = "strecha"."""
    words = norm(item).split()
    return " ".join(w[:5] for w in words)


def score_fields(pred: dict | None, gold: dict) -> dict[str, list[int]]:
    """Per field [tp, fp, fn]. A missing answer costs recall, an invented one costs precision, a wrong one both.

    Booleans are positive answers both ways: for them every field has a value to get right.
    """
    counts = {f: [0, 0, 0] for f in FIELDS}
    pred = pred or {}
    for f in FIELDS:
        p, g = pred.get(f), gold[f]
        if f == "damaged_items":
            ps = {item_key(i) for i in p} if isinstance(p, list) else set()
            gs = {item_key(i) for i in g}
            counts[f] = [len(ps & gs), len(ps - gs), len(gs - ps)]
        elif g is None:
            counts[f][1] += p is not None
        elif p is None:
            counts[f][2] += 1
        elif same_scalar(f, p, g):
            counts[f][0] += 1
        else:
            counts[f][1] += 1
            counts[f][2] += 1
    return counts


def f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def consistency_problems(text: str, rec: dict) -> list[str]:
    """Does the written report really contain the record? Checks every non-null value, and that null ones stayed out."""
    t, compact = norm(text), re.sub(r"[\s.,]", "", norm(text))
    bad = []
    if (
        rec["policy_number"]
        and norm(rec["policy_number"]) not in t
        and norm(rec["policy_number"]).replace("-", "") not in compact
    ):
        bad.append("policy_number")
    if rec["policy_number"] is None and re.search(r"\b(pz|hd|aut|maj)[- ]?\d{4,}", t):
        bad.append("policy_invented")
    if rec["city"] and not all(stem(w, 3) in t for w in rec["city"].split() if len(w) > 3):
        bad.append("city")
    if rec["estimated_amount_eur"] is not None and str(rec["estimated_amount_eur"]) not in compact:
        bad.append("amount")
    if rec["estimated_amount_eur"] is None and re.search(r"\d[\d .]*\s?(€|eur)", t):
        bad.append("amount_invented")
    months = ["janu", "febru", "marc", "april", "maj", "jun", "jul", "august", "septemb", "oktob", "novemb", "decemb"]
    if rec["event_date"] is None and re.search(rf"\b\d{{1,2}}\.\s?(\d{{1,2}}\.|{'|'.join(months)})|\b20\d\d\b", t):
        bad.append("date_invented")  # found by reading: the label said null while the text said "15. apríla 2024"
    if rec["event_date"]:
        y, m, d = rec["event_date"].split("-")
        numeric = re.search(rf"\b0?{int(d)}\.\s?0?{int(m)}\.(\s?{y})?", t) or re.search(rf"{y}-{m}-{d}", t)
        worded = re.search(rf"\b0?{int(d)}\.\s?{months[int(m) - 1]}", t)
        ordinal = re.search(rf"\b{ORDINALS[int(d)]}\s+{months[int(m) - 1]}", t)
        if not (numeric or worded or ordinal):
            bad.append("date")
        elif y not in t and not re.search(rf"\.\s?{y[2:]}\b", t):
            bad.append("date_no_year")  # found by reading: the label cannot know a year the text never states
    for item in rec["damaged_items"]:
        if not all(stem(w, 3) in t for w in item.split() if len(w) > 3):
            bad.append(f"item:{item}")
    if rec["injured_persons"] and not re.search(r"zran|ranen|ublíz|ublizen|osetr|nemocn|uraz|lekár|lekar", t):
        bad.append("injured")
    if rec["police_reported"] and not re.search(r"polic", t):
        bad.append("police")
    return bad

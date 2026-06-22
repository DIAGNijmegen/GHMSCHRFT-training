# Regenerate after environment reset
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

random.seed(777)

PLACES = [
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "Groningen",
    "Eindhoven",
    "Leiden",
    "Zwolle",
    "Nijmegen",
    "Maastricht",
    "Tilburg",
    "Haarlem",
    "Amersfoort",
    "Den Bosch",
    "Almere",
    "Dordrecht",
]
STREETS = [
    "Parkweg",
    "Dorpsstraat",
    "Stationsplein",
    "Prinsengracht",
    "Lijsterlaan",
    "Willemskade",
    "Kerkstraat",
    "Schoolstraat",
    "Molenstraat",
    "Boslaan",
    "Singel",
    "Julianalaan",
    "Burgemeesterlaan",
    "Vijzelstraat",
    "Oude Gracht",
]
COMPANIES = [
    "DiagnostiekLab Delta",
    "Medisch Centrum Noordstad",
    "Radiologie Groep Amstel",
    "Stichting Ziekenhuis Rivierenland",
    "Pathologie Service Nederland",
    "Zorggroep Stad & Ommeland",
    "Kliniek Rijnzicht",
    "Centrum voor Beeldvorming Zuid",
    "Huisartsenpraktijk De Brink",
    "Apotheek Waterstad",
    "GGD Regio Midden",
    "Thuiszorgorganisatie Zon",
]
STUDIES = [
    "NOVA-ET",
    "RADIOMICS-PANCREAS",
    "PATHO-IMMUNO-2025",
    "SEH-COVID-TRIAGE",
    "DUTCH-CTA-LUNG",
    "BIOBANK-NEPHRO",
    "CARDIO-AF-REG",
    "ONCO-CT-PILOT",
    "NEURO-STROKE-SCAN",
    "MICROBIO-UTI-TRIAL",
    "MAMMO-DENSE-2024",
    "AI-TRIAGE-NL",
]
URLS = [
    "https://onderzoek.example/studie-info",
    "https://www.zorgportaal.example/uitslag",
    "http://intranet.example.local/protocol",
    "https://trial.example.org/overview",
    "https://portal.example/aanvragen",
    "http://protocols.example.local/handleiding",
]
FIRST = [
    "Sanne",
    "Daan",
    "Noor",
    "Milan",
    "Julia",
    "Tess",
    "Jesse",
    "Eva",
    "Lars",
    "Sara",
    "Bram",
    "Lotte",
    "Sem",
    "Nina",
    "Finn",
    "Isa",
    "Thijs",
    "Fleur",
    "Olivier",
    "Zoë",
]
LAST = [
    "Jansen",
    "De Vries",
    "Van Dijk",
    "Bakker",
    "Visser",
    "Smit",
    "Meijer",
    "Mulder",
    "De Boer",
    "Bos",
    "Van Leeuwen",
    "Dekker",
    "Vos",
    "Verhoeven",
    "Peeters",
    "Willems",
]
KINDS = [
    "SEH-verslag",
    "Radiologie verslag",
    "Pathologie uitslag",
    "Ontslagbrief",
    "Verwijsbrief",
    "Microbiologie uitslag",
    "Operatieverslag",
    "Polikliniekverslag",
]


def rand_digits(n):
    return "".join(random.choice(string.digits) for _ in range(n))


def make_bsn():
    return rand_digits(9)


def make_iban():
    return f"NL{rand_digits(2)}{''.join(random.choice(string.ascii_uppercase) for _ in range(4))}{rand_digits(10)}"


def make_address(place):
    return f"{random.choice(STREETS)} {random.randint(1,240)}{random.choice(['','A','B','C','-2','bis'])}, {rand_digits(4)} {random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)} {place}"


def make_docid():
    return f"{random.randint(1,9999):04d}"


def make_docnum():
    return rand_digits(10)


def make_date(base=None):
    base = base or datetime(2018, 1, 1)
    return (base + timedelta(days=random.randint(0, 365 * 5))).strftime("%d-%m-%Y")


def make_person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def header_variants(docid, docnum, kind):
    return random.choice(
        [
            f"{docid} ; {docnum} ; {kind}:\n",
            f"{docid};{docnum};{kind}\n",
            f"Nr {docid} / dossier {docnum} - {kind}\n",
            f"{kind} (document {docid}, nummer {docnum})\n",
        ]
    )


def build_text(vals):
    kind = random.choice(KINDS)
    text = f"""{header_variants(vals['docid'], vals['docnum'], kind)}
Datum: {vals['datum1']}

Locatie/plaats: {vals['place']}
Behandelaar: dr. {vals['persoon']}

ADMINISTRATIE
BSN: {vals['bsn']}
Adres: {vals['adres']}
Verzekeraar/bedrijf: {vals['bedrijf']}
IBAN: {vals['iban']}
Studie: {vals['studie']}
URL: {vals['url']}

KLINIEK
Patiënt gezien op {vals['datum2']} i.v.m. klachten. Gevallen in {vals['place']} tijdens vakantie.
Contact via {vals['bedrijf']} en website {vals['url']}.
"""
    if random.random() < 0.5:
        text += f"\nAanvullend: overleg met dr. {vals['persoon']} op {vals['datum1']}."
    if random.random() < 0.4:
        text += f"\nAdministratief bevestigd: {vals['adres']} | {vals['iban']}."
    return text


def find_all(text, sub):
    out = []
    start = 0
    while True:
        i = text.find(sub, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def build_meta_and_bad(text, vals):
    entries = [
        ("<DOCUMENTID>", vals["docid"]),
        ("<DOCUMENTNUMMER>", vals["docnum"]),
        ("<DATUM>", vals["datum1"]),
        ("<DATUM>", vals["datum2"]),
        ("<PERSOON>", vals["persoon"]),
        ("<PLAATS>", vals["place"]),
        ("<BSN>", vals["bsn"]),
        ("<ADRES>", vals["adres"]),
        ("<BEDRIJF>", vals["bedrijf"]),
        ("<IBAN>", vals["iban"]),
        ("<STUDIE_NAAM>", vals["studie"]),
        ("<URL>", vals["url"]),
    ]
    meta = []
    bad = []
    for tag, val in entries:
        occ = find_all(text, val)
        if not occ:
            raise ValueError(f"Missing {tag}:{val}")
        true = random.choice(occ)
        old = max(0, min(len(text) - 1, true + random.randint(-8, 8)))
        bad_start = old
        bad_len = len(val) + (
            random.choice([-2, -1, 0, 1, 2])
            if len(val) > 4 and random.random() < 0.25
            else 0
        )
        bad_end = max(bad_start, min(len(text), bad_start + bad_len))
        bad.append([bad_start, bad_end, tag])
        meta.append({"tag": tag, "value": val, "old_start": old, "true_start": true})
    bad.sort(key=lambda x: (x[0], x[1], x[2]))
    return meta, bad


def make_record(i):
    vals = {
        "docid": make_docid(),
        "docnum": make_docnum(),
        "datum1": make_date(),
        "datum2": make_date(),
        "persoon": make_person(),
        "place": random.choice(PLACES),
        "bsn": make_bsn(),
        "adres": None,
        "bedrijf": random.choice(COMPANIES),
        "iban": make_iban(),
        "studie": random.choice(STUDIES),
        "url": random.choice(URLS),
    }
    vals["adres"] = make_address(vals["place"])
    text = build_text(vals)
    meta, bad = build_meta_and_bad(text, vals)
    return {
        "id": 830000 + i,
        "text": text,
        "uid": int(vals["docid"]),
        "Comments": [],
        "label": bad,
        "meta_entities": meta,
    }


out = Path("synthetic_with_meta.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i in range(50):
        f.write(json.dumps(make_record(i + 1), ensure_ascii=False) + "\n")

(out.as_posix(), out.stat().st_size)

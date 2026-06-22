#!/usr/bin/env python3
"""
Generate diverse Dutch medical reports via a remote Ollama server.

LLM returns STRICT JSON with:
- "text": report containing inline tags, e.g. <DATUM>...</DATUM>
- "entities": {} (ignored; spans are derived from tags)

We then:
1) Parse and extract spans from inline tags (ALL occurrences)
2) Remove the tags from the text
3) Build meta_entities (true spans) + label spans (noisy spans)

Diversity measures:
- higher temperature/top_p
- random report type sampling (SEH / Radiologie / Pathologie / etc.)
- automatic rejection & retry if the model uses common placeholder values
  like "Mevrouw A. de Vries" or BSN "123456789"
"""

import argparse
import json
import random
import time
from typing import Any, Dict, List, Tuple

import requests

TAG_NAMES: List[str] = [
    "DATUM",
    "PERSOON",
    "BSN",
    "ADRES",
    "BEDRIJF",
    "ZIEKENHUIS",
    "IBAN",
    "STUDIE_NAAM",
    "URL",
]
TAGS_BRACKETED: List[str] = [f"<{t}>" for t in TAG_NAMES]

REPORT_TYPES: List[Tuple[str, str]] = [
    (
        "SEH-verslag",
        "SEH / spoedeisende hulp: acuut consult, triage, vitale parameters, beoordeling en beleid.",
    ),
    (
        "Radiologie verslag",
        "Beeldvorming (CT/MRI/echo/röntgen): indicatie, techniek, bevindingen, conclusie/advies.",
    ),
    (
        "Pathologie uitslag",
        "Histologie/cytologie: macroscopie, microscopie, (immuno)kleuringen, conclusie/differentiaal.",
    ),
    (
        "Ontslagbrief",
        "Klinische opname: samenvatting opname, diagnoses, behandeling, medicatie, follow-up.",
    ),
    (
        "Polikliniekverslag",
        "Polikliniek: anamnese, bevindingen, interpretatie, plan, vervolgafspraken.",
    ),
]

# Common placeholders to avoid
BAD_PERSON_VALUES = {
    "Mevrouw A. de Vries",
    "Mevrouw A. De Vries",
    "A. de Vries",
    "A. De Vries",
    "Mevrouw De Vries",
    "Dhr. Jansen",
    "Jan Jansen",
}
BAD_BSN_VALUES = {
    "123456789",
    "111111111",
    "000000000",
    "999999999",
}


def make_prompt(
    report_type: str,
    style_hint: str,
    style_seed: str,
    avoid: Dict[str, str] | None = None,
) -> str:
    avoid_lines = ""
    if avoid:
        # Keep this extremely short and specific.
        # We only mention values we observed, to discourage repeating.
        avoid_lines = (
            "\nVermijd exact deze waarden in de tags: "
            + ", ".join(f"<{k}>{v}</{k}>" for k, v in avoid.items())
            + "\n"
        )

    return f"""
Schrijf een realistisch Nederlands {report_type} (vrije stijl, 250–900 woorden).
Focus: {style_hint}
Schrijfstijl-variatie: {style_seed}

Je MOET ALLE voorkomens van de volgende gegevens in de TEKST taggen door ze inline te omringen met tags, bijvoorbeeld:
<DATUM>12-03-2021</DATUM> en <PERSOON>Mevrouw A. de Vries</PERSOON>

Te gebruiken tags:
{", ".join(f"<{t}>...</{t}>" for t in TAG_NAMES)}

Inhoudsregels:
- <ZIEKENHUIS> is een ziekenhuisnaam.
- <BEDRIJF> is een werkgever/organisatie (geen ziekenhuis).
- Gebruik geen verklarende tekst direct achter entiteiten zoals "(STUDIE NAAM)" of "(BSN)" of "studie:" naast de waarde.
- Vermijd voorbeeld-placeholders zoals "Mevrouw A. de Vries" en BSN "123456789"; verzin steeds nieuwe, gevarieerde waarden.
{avoid_lines}
Output:
Geef je antwoord ALS STRICTE JSON met precies twee velden:
{{
  "text": "de volledige verslagtekst met inline tags",
  "entities": {{}}
}}

Alleen JSON. Geen markdown. Geen uitleg. Geen extra velden.
""".strip()


def ollama_chat_json(
    host: str, model: str, prompt: str, timeout_s: float = 1200.0
) -> Dict[str, Any]:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 1.15,
            "top_p": 0.97,
            "repeat_penalty": 1.12,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    raw = (data.get("message") or {}).get("content") or ""
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty response from model")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError(f"Model did not return JSON. Raw start: {raw[:200]!r}")


def parse_and_strip_inline_tags(
    tagged_text: str, rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], List[List[Any]]]:
    open_tags = {t: f"<{t}>" for t in TAG_NAMES}
    close_tags = {t: f"</{t}>" for t in TAG_NAMES}

    i = 0
    clean_chars: List[str] = []
    meta_entities: List[Dict[str, Any]] = []
    labels: List[List[Any]] = []

    def match_open(pos: int) -> str | None:
        if pos >= len(tagged_text) or tagged_text[pos] != "<":
            return None
        for t in TAG_NAMES:
            if tagged_text.startswith(open_tags[t], pos):
                return t
        return None

    # A small optimization: avoid repeated join in loop by tracking current clean length.
    clean_len = 0

    while i < len(tagged_text):
        t = match_open(i)
        if t is None:
            clean_chars.append(tagged_text[i])
            clean_len += 1
            i += 1
            continue

        i += len(open_tags[t])
        close = close_tags[t]
        j = tagged_text.find(close, i)
        if j == -1:
            raise ValueError(f"Missing closing tag {close} for <{t}>")

        value = tagged_text[i:j]

        true_start = clean_len
        clean_chars.append(value)
        clean_len += len(value)
        true_end = clean_len

        tag_br = f"<{t}>"

        old_start = max(0, true_start + rng.randint(-8, 8))
        old_start = min(old_start, true_start)

        bad_start = max(0, min(true_start, old_start))
        bad_len = len(value) + (
            rng.choice([-2, -1, 0, 1, 2])
            if len(value) > 4 and rng.random() < 0.25
            else 0
        )
        bad_end = max(bad_start, min(true_end, bad_start + bad_len))

        meta_entities.append(
            {
                "tag": tag_br,
                "value": value,
                "true_start": true_start,
                "true_end": true_end,
                "old_start": old_start,
            }
        )
        labels.append([bad_start, bad_end, tag_br])

        i = j + len(close)

    clean_text = "".join(clean_chars)

    present = {m["tag"] for m in meta_entities}
    missing = [t for t in TAGS_BRACKETED if t not in present]
    if missing:
        raise ValueError(f"Missing at least one required tag occurrence: {missing}")

    labels.sort(key=lambda x: (x[0], x[1], x[2]))
    meta_entities.sort(key=lambda x: (x["true_start"], x["true_end"], x["tag"]))
    return clean_text, meta_entities, labels


def extract_values(meta_entities: List[Dict[str, Any]], tag_br: str) -> List[str]:
    return [m["value"] for m in meta_entities if m["tag"] == tag_br]


def is_bad_bsn(bsn: str) -> bool:
    b = "".join(ch for ch in bsn if ch.isdigit())
    if len(b) != 9:
        return True
    if b in BAD_BSN_VALUES:
        return True
    # reject trivial sequences/repeats
    if b == b[0] * 9:
        return True
    if b in ("123456789", "987654321"):
        return True
    return False


def is_bad_person(name: str) -> bool:
    n = " ".join(name.split())
    if n in BAD_PERSON_VALUES:
        return True
    # Too generic fallback patterns
    if n.lower() in {"mevrouw a. de vries", "dhr. jansen"}:
        return True
    # very short names like "A. de Vries"
    if len(n) <= 10 and "." in n:
        return True
    return False


def generate_one(
    host: str, model: str, rng: random.Random, max_retries: int = 16
) -> Dict[str, Any]:
    report_type, style_hint = rng.choice(REPORT_TYPES)
    style_seed = rng.choice(
        [
            "Gebruik duidelijke kopjes, maar varieer de volgorde en lengte.",
            "Schrijf vooral in alinea's met enkele korte opsommingen.",
            "Maak het compact en zakelijk, met korte zinnen en een duidelijke conclusie.",
            "Gebruik een narratievere stijl en wissel af met puntsgewijze bevindingen.",
            "Gebruik korte regels voor bevindingen en een aparte conclusie-regel.",
        ]
    )

    # If we detect placeholders, we can pass a tiny 'avoid' dict next attempt.
    avoid: Dict[str, str] | None = None
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            prompt = make_prompt(report_type, style_hint, style_seed, avoid=avoid)
            obj = ollama_chat_json(host, model, prompt)

            tagged_text = obj.get("text")
            if not isinstance(tagged_text, str) or len(tagged_text) < 250:
                raise ValueError("Invalid/short 'text'")

            clean_text, meta_entities, labels = parse_and_strip_inline_tags(
                tagged_text, rng
            )

            # Diversity guardrails (post-check)
            persons = extract_values(meta_entities, "<PERSOON>")
            bsns = extract_values(meta_entities, "<BSN>")

            if any(is_bad_person(p) for p in persons) or any(
                is_bad_bsn(b) for b in bsns
            ):
                # Prepare a very small avoid-hint using the first offending values
                bad_p = next((p for p in persons if is_bad_person(p)), None)
                bad_b = next((b for b in bsns if is_bad_bsn(b)), None)
                avoid = {}
                if bad_p:
                    avoid["PERSOON"] = bad_p
                if bad_b:
                    avoid["BSN"] = bad_b
                # Retry
                raise ValueError(f"Rejected placeholders (avoid={avoid})")

            return {
                "report_type": report_type,
                "text": clean_text,  # tag-free
                "meta_entities": meta_entities,
                "label": labels,
            }

        except Exception as e:
            last_err = e
            time.sleep(min(12.0, 1.5 * attempt))

    raise RuntimeError(f"Failed after retries. Last error: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="e.g. http://10.21.4.206:11434")
    ap.add_argument("--model", required=True, help="e.g. qwen3:30b")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="synthetic_with_meta_llm.jsonl")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(1, args.n + 1):
            g = generate_one(args.host, args.model, rng)

            rec = {
                "id": 830000 + i,
                "text": g["text"],
                "uid": 830000 + i,
                "Comments": [],
                "label": g["label"],
                "meta_entities": g["meta_entities"],
                "report_type": g["report_type"],
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i:02d}/{args.n}] ok [{g['report_type']}]")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

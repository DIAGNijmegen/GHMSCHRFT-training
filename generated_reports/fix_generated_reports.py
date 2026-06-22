import json
from pathlib import Path


def find_all(text: str, sub: str):
    out = []
    start = 0
    while True:
        i = text.find(sub, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def dedupe_spans(spans):
    # spans: list of (start,end,tag)
    seen = set()
    out = []
    for s, e, t in spans:
        key = (s, e, t)
        if key not in seen:
            seen.add(key)
            out.append((s, e, t))
    return out


def drop_overlaps(spans):
    # keep earliest, drop later overlaps (simple policy)
    spans = sorted(spans, key=lambda x: (x[0], x[1], x[2]))
    kept = []
    for s, e, t in spans:
        if not kept:
            kept.append((s, e, t))
            continue
        ps, pe, pt = kept[-1]
        if s < pe:  # overlap
            # Prefer longer span; tie -> keep existing
            if (e - s) > (pe - ps):
                kept[-1] = (s, e, t)
            else:
                continue
        else:
            kept.append((s, e, t))
    return kept


def fix_jsonl_all_occurrences(
    in_path: str, out_path: str, remove_overlaps=True, strip_meta=False
):
    inp = Path(in_path)
    outp = Path(out_path)

    total = 0
    fixed = 0

    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            text = rec.get("text", "")
            meta = rec.get("meta_entities", [])

            if not isinstance(text, str) or not isinstance(meta, list) or not meta:
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            spans = []
            for ent in meta:
                tag = ent["tag"]
                val = ent["value"]
                for start in find_all(text, val):
                    spans.append((start, start + len(val), tag))

            spans = dedupe_spans(spans)
            if remove_overlaps:
                spans = drop_overlaps(spans)

            rec["label"] = [[s, e, t] for (s, e, t) in spans]
            if strip_meta:
                rec.pop("meta_entities", None)

            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fixed += 1

    print(
        f"Processed {total} records. Rebuilt labels for {fixed} records -> {out_path}"
    )


if __name__ == "__main__":
    fix_jsonl_all_occurrences(
        "synthetic_with_meta_llm.jsonl",
        "synthetic_fixed.jsonl",
        remove_overlaps=True,
        strip_meta=False,
    )

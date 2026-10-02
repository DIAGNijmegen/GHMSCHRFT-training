"""
Audit the reader-study spans that were reclassified as effectively de-identified,
and emit a supplementary table that contains no clinical text.

The reader study scored only spans a human reader could realistically recognise.
evaluate_readers.py holds that restriction as two hard-coded lists: EXCLUDED_CASES
(32 whole cases) and EXCLUDED_SPANS (2 spans inside one retained case). Those
lists record the decision but not the evidence, which is what a reviewer asks for.

For every reclassified span this script recovers, from reader_cases.jsonl alone:

  * the PHI tag of the annotated gold span
  * how many tokens the span had, and how many of them HIPS replaced
  * how many characters survived replacement

None of that is clinical text, and together it is the argument: HIPS replaced most
of each span and what survived is short. The Reason column comes from
reclassified_reasons.json, passed with --reasons, which records why each remainder
was judged not to identify a patient. Without that file the column falls back to a
computed description of the fragment by character class.

Pass --show-text to print the surviving fragments for your own inspection. That
output is residual content from clinical reports and is NOT safe to export.

Usage (PowerShell, from the repository root):

    python evaluation/readerstudy/summarise_reclassified.py `
        --output-dir evaluation\\readerstudy\\output `
        --reasons evaluation\\readerstudy\\reclassified_reasons.json `
        --latex-out reclassified_table.tex
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

# Kept in sync with evaluation/readerstudy/evaluate_readers.py. Change both together.
EXCLUDED_CASES = frozenset({
    6, 8, 9, 12, 17, 19, 20, 24, 27, 28, 30, 31, 32, 36,
    46, 47, 49, 52, 54, 58, 63, 66, 67, 70,
    83, 84, 85, 86, 96, 97, 98, 100,
})
EXCLUDED_SPANS = {
    74: [(227, 239), (2210, 2217)],
}

# Dutch function words, honorifics and unit words that carry nothing
# patient-specific on their own.
NON_IDENTIFYING_WORDS = {
    "jaar", "jr", "jrg", "maand", "maanden", "week", "weken", "dag", "dagen",
    "uur", "min", "sec", "dhr", "mevr", "mw", "hr", "dr", "drs", "prof", "ir",
    "ing", "mr", "de", "het", "een", "van", "der", "den", "te", "op", "om",
    "in", "bij", "aan", "tot", "en", "nr", "no", "nummer", "ziekenhuis",
    "afdeling", "polikliniek", "poli",
}

BLANK = ""


def describe_fragment(fragment: str) -> str:
    """
    Describe what survived replacement, by character class only.

    This is computed, not judged. Every row gets a value, nothing is left for a
    human to fill in, and no report content is reproduced. A reader combining
    this with the token counts can see how much of each annotated span the
    surrogate module rewrote and what kind of residue was left.
    """
    stripped = fragment.strip()
    if not stripped:
        return "nothing visible"
    if not any(c.isalnum() for c in stripped):
        return "punctuation or separator only"

    has_digit = any(c.isdigit() for c in stripped)
    has_alpha = any(c.isalpha() for c in stripped)
    words = len(stripped.split())
    if has_digit and not has_alpha:
        n = sum(1 for c in stripped if c.isdigit())
        return "single digit" if n == 1 else "digits only"
    if has_alpha and not has_digit:
        return "one word, letters only" if words == 1 else f"{words} words, letters only"
    return "letters and digits"


def load_jsonl(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["case_index"]] = r
    return records


def token_starts_for(tokens):
    starts, pos = [], 0
    for tok in tokens:
        starts.append(pos)
        pos += len(tok) + 1
    return starts


def missed_spans(case):
    """
    Recompute the missed gold spans exactly as prepare_readerstudy.find_errors
    does: a gold span is missed when any of its tokens was predicted O.

    Yields (char_start, char_end, tag, n_tokens, surviving_tokens).
    """
    tokens = case["text_parts"]
    true_labels = case["named_entity_recognition_target"]
    pred_labels = case["docker_pred_labels"]
    starts = token_starts_for(tokens)

    i = 0
    while i < len(true_labels):
        if not true_labels[i].startswith("B-"):
            i += 1
            continue
        tag = true_labels[i][2:]
        j = i + 1
        while j < len(true_labels) and true_labels[j] == f"I-{tag}":
            j += 1
        if any(pred_labels[k] == "O" for k in range(i, j)):
            surviving = [tokens[k] for k in range(i, j) if pred_labels[k] == "O"]
            yield (starts[i], starts[j - 1] + len(tokens[j - 1]),
                   tag, j - i, surviving)
        i = j


def clean_tag(tag: str) -> str:
    return tag.strip("<>").upper()


def latex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def collect(cases):
    rows = []
    for idx in sorted(EXCLUDED_CASES):
        case = cases.get(idx)
        if case is None:
            print(f"  warning: case {idx} not in reader_cases.jsonl", file=sys.stderr)
            continue
        for order, (cs, ce, tag, n_tok, surviving) in enumerate(missed_spans(case), 1):
            rows.append(make_row(idx, order, "case", tag, n_tok, surviving))

    for idx, spans in sorted(EXCLUDED_SPANS.items()):
        case = cases.get(idx)
        if case is None:
            print(f"  warning: case {idx} not in reader_cases.jsonl", file=sys.stderr)
            continue
        wanted = {(s, e) for s, e in spans}
        for order, (cs, ce, tag, n_tok, surviving) in enumerate(missed_spans(case), 1):
            if (cs, ce) in wanted:
                rows.append(make_row(idx, order, "span", tag, n_tok, surviving))
    return rows


def make_row(idx, order, kind, tag, n_tok, surviving):
    fragment = " ".join(surviving)
    return {
        "key": f"{idx}.{order}",
        "case": idx,
        "kind": kind,
        "tag": clean_tag(tag),
        "tokens_total": n_tok,
        "tokens_replaced": n_tok - len(surviving),
        "chars_surviving": len(fragment.strip()),
        "reason": describe_fragment(fragment),
        "fragment": fragment,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Reader-study output directory holding reader_cases.jsonl")
    ap.add_argument("--latex-out", type=Path, default=None,
                    help="Write the supplementary longtable here")
    ap.add_argument("--reasons", type=Path, default=None,
                    help='JSON file of {"<case>.<n>": "text"} overriding the '
                         'computed description for individual spans')
    ap.add_argument("--show-text", action="store_true",
                    help="Also print the surviving fragments. NOT safe to export.")
    args = ap.parse_args()

    cases_file = args.output_dir / "reader_cases.jsonl"
    if not cases_file.exists():
        sys.exit(f"reader_cases.jsonl not found at {cases_file}")
    cases = load_jsonl(cases_file)

    rows = collect(cases)

    if args.reasons and args.reasons.exists():
        manual = json.loads(args.reasons.read_text(encoding="utf-8"))
        unknown = [k for k in manual if k not in {r["key"] for r in rows}]
        if unknown:
            print(f"  warning: reasons file has keys matching no span: {unknown}",
                  file=sys.stderr)
        for r in rows:
            if manual.get(r["key"]):
                r["reason"] = manual[r["key"]]

    hdr = (f"{'Span':>7}  {'Kind':<5} {'PHI tag':<18} {'Tok':>4} {'Repl':>5} "
           f"{'Chars':>6}  Reason")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['key']:>7}  {r['kind']:<5} {r['tag']:<18} "
                f"{r['tokens_total']:>4} {r['tokens_replaced']:>5} "
                f"{r['chars_surviving']:>6}  {r['reason']}")
        if args.show_text:
            line += f"   |{r['fragment']}|"
        print(line)

    n_case_rows = sum(1 for r in rows if r["kind"] == "case")
    n_span_rows = sum(1 for r in rows if r["kind"] == "span")
    print(f"\n{len(rows)} reclassified spans: {n_case_rows} in "
          f"{len({r['case'] for r in rows if r['kind'] == 'case'})} excluded cases, "
          f"{n_span_rows} excluded individually.")

    chars = [r["chars_surviving"] for r in rows]
    frac = [r["tokens_replaced"] / r["tokens_total"] for r in rows]
    print(f"Tokens replaced by HIPS: median "
          f"{median(frac) * 100:.0f}% of the annotated span "
          f"(range {min(frac) * 100:.0f} to {max(frac) * 100:.0f}%).")
    print(f"Characters surviving: median {median(chars):.0f}, "
          f"range {min(chars)} to {max(chars)}; "
          f"{sum(1 for c in chars if c <= 8)} of {len(chars)} are 8 or fewer.")

    print("\nBy PHI tag:")
    by_tag = Counter(r["tag"] for r in rows)
    for tag, n in sorted(by_tag.items()):
        print(f"  {n:>4}  {tag}")

    print("\nBy reason:")
    for reason, n in Counter(r["reason"] for r in rows).most_common():
        print(f"  {n:>4}  {reason}")

    todo = [r for r in rows if not r["reason"]]
    if todo:
        print(f"\n{len(todo)} span(s) have no description, which should not "
              f"happen: {[r['key'] for r in todo]}")
    else:
        print("\nEvery reclassified span has a computed description.")

    if args.latex_out:
        write_latex(rows, args.latex_out)
        print(f"\nLaTeX table written to {args.latex_out}")


def write_latex(rows, path: Path):
    out = [
        r"%% GENERATED by evaluation/readerstudy/summarise_reclassified.py in",
        r"%% GHMSCHRFT-training-DIAG. Do not edit by hand; regenerate instead.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{longtable}{@{}l l l r r r >{\raggedright\arraybackslash}p{4.4cm}@{}}",
        r"\caption{Every reader-study span reclassified as effectively "
        r"de-identified. Span is the reader-study case index and the index of "
        r"the missed span within that case. Scope records whether the whole "
        r"case was set aside or the span alone. Tokens is the length of the "
        r"annotated gold span and Replaced is how many of its tokens the "
        r"surrogate module rewrote. Chars is the length in characters of what "
        r"escaped replacement, and Reason records why that remainder was "
        r"judged not to identify a patient. The fragments themselves are "
        r"not reproduced, "
        r"because they are residual content from clinical reports.}",
        r"\label{tab:reclassified_spans} \\",
        r"\toprule",
        r"\textbf{Span} & \textbf{Scope} & \textbf{PHI tag} & \textbf{Tokens} & "
        r"\textbf{Replaced} & \textbf{Chars} & \textbf{Reason} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Span} & \textbf{Scope} & \textbf{PHI tag} & \textbf{Tokens} & "
        r"\textbf{Replaced} & \textbf{Chars} & \textbf{Reason} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for r in rows:
        scope = "case" if r["kind"] == "case" else "span"
        reason = r["reason"]
        out.append(
            f"{r['key']} & {scope} & \\tagname{{{latex_escape(r['tag'])}}} & "
            f"{r['tokens_total']} & {r['tokens_replaced']} & "
            f"{r['chars_surviving']} & {latex_escape(reason)} \\\\")
    out.append(r"\end{longtable}")
    out.append(r"\endgroup")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

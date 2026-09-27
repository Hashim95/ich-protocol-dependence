#!/usr/bin/env python3
"""
make_survey_audit.py — build results/tables/table1_survey_audit.csv from the
survey table in main.tex.

WHY THIS EXISTS
---------------
The Table 1 prose counts ("six of fourteen studies state patient-level
partitioning") were being written by hand and drifted from the table itself.
emit_macros() in 11_build_tables.py derives \surveyN, \surveyStated, \surveyCI
and \surveyNoEDH from a CSV, so the CSV has to exist and has to match the table.

This script builds it FROM main.tex rather than from a separate list, so the
two cannot disagree. It never invents a row: if parsing fails it says so and
writes nothing.

A previous attempt at this file was hand-written from memory and contained
fourteen studies that are not in the verified audit (including four attributed
to MIMIC-III, which contains no head CT imaging). Deriving from main.tex makes
that failure mode impossible.

USAGE
    python make_survey_audit.py                 # auto-find main.tex
    python make_survey_audit.py --tex path.tex  # explicit path
    python make_survey_audit.py --dry-run       # show rows, write nothing
"""
import argparse, csv, os, re, sys
from pathlib import Path

# The fourteen studies that should be present. Used as a sanity check only —
# the values come from main.tex, this list just confirms we parsed the right
# table and did not silently pick up a stale copy of the file.
EXPECTED_AUTHORS = {
    "chang", "ye", "lee", "wang", "angkurawaranon", "kang", "yeo",
    "zhang", "chetla", "liu", "chaudhary", "chagahi", "abrigo",
}

HEADER = ["Study", "Data", "Level", "EDH metric",
          "Patient-disjoint", "External", "CI"]

# emit_macros() matches on these exact strings. If the table wording changes,
# the counts silently go wrong, so we verify them here instead.
PD_VALID = ("Stated", "Not stated", "No", "n/a")
NOEDH_MARKERS = ("pooled", "macro-averaged", "not reported")


def find_tex(explicit=None):
    """Locate main.tex, preferring the most recently modified copy."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            sys.exit(f"not found: {p}")
        return p
    home = Path.home()
    cands = [home / "Downloads" / "main.tex",
             home / "paper1_overleaf" / "main.tex",
             home / "paper1" / "main.tex",
             Path.cwd() / "main.tex"]
    cands += list(home.glob("Downloads/**/main.tex"))
    found = [c for c in cands if c.exists()]
    if not found:
        sys.exit("could not find main.tex — pass --tex /path/to/main.tex")
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(found) > 1:
        print("multiple copies found, using the most recent:")
        for c in found:
            import datetime
            t = datetime.datetime.fromtimestamp(c.stat().st_mtime)
            print(f"   {'->' if c == found[0] else '  '} {t:%Y-%m-%d %H:%M}  {c}")
    return found[0]


def clean(cell):
    """Strip LaTeX markup from one cell, leaving the plain value."""
    c = cell
    c = re.sub(r"~?\\cite\{[^}]*\}", "", c)          # citations
    c = re.sub(r"\$\^\{?[^}$]*\}?\$", "", c)         # footnote daggers
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)     # bold
    c = re.sub(r"\\emph\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\texttt\{([^}]*)\}", r"\1", c)
    c = c.replace(r"\,", " ").replace(r"$\rightarrow$", "->")
    c = re.sub(r"\\[a-zA-Z]+", "", c)                # any remaining commands
    c = c.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", c).strip()


def extract(tex_path):
    """Pull the survey table body out of main.tex and split it into rows.

    Handles rows that wrap across source lines by joining the whole block and
    splitting on the LaTeX row terminator instead of on newlines.
    """
    src = tex_path.read_text(errors="ignore")
    if r"\label{tab:survey}" not in src:
        sys.exit(f"no \\label{{tab:survey}} in {tex_path}\n"
                 f"This is probably an older copy without the verified table.")
    body = src.split(r"\label{tab:survey}", 1)[1]
    if r"\bottomrule" not in body:
        sys.exit("found tab:survey but no \\bottomrule — table looks malformed")
    body = body.split(r"\bottomrule", 1)[0]
    # keep only what is between the first \midrule and the end
    if r"\midrule" in body:
        body = body.split(r"\midrule", 1)[1]
    body = re.sub(r"%.*", "", body)                  # drop comments
    body = body.replace(r"\midrule", " ")            # the pre-"This work" rule

    rows, bad = [], []
    for raw in body.split(r"\\"):
        if "&" not in raw:
            continue
        cells = [clean(c) for c in raw.split("&")]
        if len(cells) != len(HEADER):
            if any(cells):
                bad.append((len(cells), " | ".join(cells)[:90]))
            continue
        if not cells[0]:
            continue
        rows.append(cells)
    return rows, bad


def validate(rows):
    """Check the parse produced the verified table, not something else."""
    problems = []
    prior = [r for r in rows if "this work" not in r[0].lower()]

    got = set()
    for r in prior:
        m = re.match(r"([A-Za-z]+)", r[0])
        if m:
            got.add(m.group(1).lower())
    missing = EXPECTED_AUTHORS - got
    unexpected = got - EXPECTED_AUTHORS
    if missing:
        problems.append(f"expected authors absent from the parse: {sorted(missing)}")
    if unexpected:
        problems.append(f"unrecognised authors present: {sorted(unexpected)} "
                        f"(verify these are real rows before proceeding)")

    for r in prior:
        if not any(r[4].startswith(v) for v in PD_VALID):
            problems.append(f"'{r[0]}' patient-disjoint = '{r[4]}' "
                            f"— must start with one of {PD_VALID}")
        if r[6] and not r[6].startswith(("Yes", "No", "n/a")):
            problems.append(f"'{r[0]}' CI = '{r[6]}' — must start Yes/No/n/a")

    counts = dict(
        n=len(prior),
        stated=sum(r[4].startswith("Stated") for r in prior),
        not_stated=sum(r[4].startswith("Not stated") for r in prior),
        ci=sum(r[6].startswith("Yes") for r in prior),
        no_edh=sum(any(m in r[3].lower() for m in NOEDH_MARKERS) for r in prior),
    )
    return problems, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tex = find_tex(a.tex)
    print(f"\nreading {tex}\n")

    rows, bad = extract(tex)
    if bad:
        print(f"[warn] {len(bad)} line(s) had the wrong column count and were skipped:")
        for n, s in bad[:5]:
            print(f"   {n} cells: {s}")
        print()
    if not rows:
        sys.exit("parsed zero rows — the table format is not what this script "
                 "expects. Write the CSV by hand from the table in main.tex; "
                 "it is only fourteen lines.")

    print(f"{len(rows)} rows parsed:\n")
    print(f"  {'Study':<28}{'Patient-disjoint':<14}{'CI':<6}EDH metric")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r[0][:27]:<28}{r[4][:13]:<14}{r[6][:5]:<6}{r[3][:30]}")

    problems, c = validate(rows)
    print(f"\nderived counts (prior studies only, excluding 'This work'):")
    print(f"  \\surveyN         = {c['n']}")
    print(f"  \\surveyStated    = {c['stated']}")
    print(f"  \\surveyNotStated = {c['not_stated']}")
    print(f"  \\surveyCI        = {c['ci']}")
    print(f"  \\surveyNoEDH     = {c['no_edh']}")

    if problems:
        print("\nPROBLEMS — resolve before using these counts:")
        for p in problems:
            print(f"  - {p}")
        print("\nNothing written. Fix main.tex (or pass --tex for the right copy)"
              " and re-run.")
        sys.exit(1)

    print("\nvalidation passed: authors match the verified audit, all "
          "patient-disjoint and CI cells use recognised wording.")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = a.out or os.path.expandvars(
        "$ICH_ROOT/results/tables/table1_survey_audit.csv")
    if out.startswith("$"):
        sys.exit("ICH_ROOT is not set — export it, or pass --out")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n[written] {out}")
    print("\nnext:  python 11_build_tables.py --macros-only")
    print("       grep survey \"$ICH_ROOT/results/tables/numbers.tex\"")


if __name__ == "__main__":
    main()
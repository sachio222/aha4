"""Builds the README's result tables from the JSON files in results/.

Every number in the README comes from a file that evaluate.py wrote. This
script reads them, prints the tables as markdown, and writes:

    results/tables.md   the markdown tables
    results/grid.csv    one line per JSON file, every stage's mean and std

Example:
    python scripts/make_tables.py
"""

import argparse
import csv
import json
from pathlib import Path

STAGES = ("pixels", "vc", "pr", "pc", "pm")
PAPER = {"vc": 0.716, "pr": 0.864}

# (file, group, label). The order here is the order in the tables. Files in
# results/ that aren't listed go at the end of the grid under their name.
ROWS = [
    ("oneshot.json", "grid", "Default: study-to-test, raw scale"),
    ("match_test_to_study.json", "grid", "Test-to-study matching, the repo's first version"),
    ("normalise.json", "grid", "Normalised matching"),
    ("normalise_test_to_study.json", "grid", "Normalised matching, test-to-study"),
    ("normalise_input.json", "grid", "VC output normalised before DG and PR"),
    ("normalise_input_normalise.json", "grid",
     "VC output normalised before DG and PR, normalised matching"),
    ("pretrain_seeds.json", "grid", "Default, VC pretrained with seeds 0, 1 and 2"),
    ("mse.json", "grid", "MSE pretraining"),
    ("mse_normalise.json", "grid", "MSE pretraining, normalised matching"),
    ("mse_no_interest_filter.json", "grid", "MSE pretraining, no interest filter"),
    ("mse_normalise_input_normalise.json", "grid",
     "MSE pretraining, VC output normalised before DG and PR, normalised matching"),
    ("best.json", "grid", "Best configuration found, picked on test scores"),
    ("best_pretrain_seeds.json", "grid",
     "Best configuration found, VC pretrained with seeds 0, 1 and 2"),
    ("ablation_no_interest_filter.json", "ablation", "No interest filter"),
    ("ablation_no_centre.json", "ablation", "No centring of drawings"),
    ("ablation_hebbian.json", "ablation", "Hebbian learning in CA3 instead of pseudo-inverse"),
    ("cpu_seed0.json", "device", "CPU, seed 0"),
]


def load_rows(results_dir="results"):
    """Reads every results JSON, in ROWS order, then any unlisted ones."""
    results_dir = Path(results_dir)
    known = {name: (group, label) for name, group, label in ROWS}
    files = [name for name, _, _ in ROWS if (results_dir / name).is_file()]
    files += sorted(p.name for p in results_dir.glob("*.json") if p.name not in known)

    rows = []
    for name in files:
        record = json.loads((results_dir / name).read_text())
        group, label = known.get(name, ("grid", Path(name).stem))
        rows.append({"file": name, "group": group, "label": label, **record})
    return rows


def pretrain_loss(row):
    losses = {p.get("loss", "bce") for p in row.get("pretrain", [{}])}
    return "/".join(sorted(losses))


def settings(row):
    """The options a row was run with, as a dict of short strings."""
    args = row.get("args", {})
    checkpoints = args.get("checkpoint", [])
    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]
    return {
        "match": args.get("match", "test-to-study"),
        "normalise": bool(args.get("normalise", False)),
        "normalise_input": bool(args.get("normalise_input", False)),
        "loss": pretrain_loss(row),
        "interest_filter": not args.get("no_interest_filter", False),
        "centre": not args.get("no_centre", False),
        "ca3_rule": args.get("ca3_rule", "pinverse"),
        "checkpoints": len(checkpoints),
        "seeds": args.get("seeds", ""),
        "spread": row.get("spread", "seeds"),
    }


def cell(row, stage):
    s = row["summary"].get(stage)
    if s is None:
        return "-"
    if s["std"] > 0:
        return "{:.1%} ± {:.1f}".format(s["mean"], 100 * s["std"])
    return "{:.1%}".format(s["mean"])


def grid_table(rows):
    """Markdown table: one line per row, VC to PM."""
    lines = ["| Setting | Matching | VC(EC) | PR(EC-CA3) | PC(CA3) | PM(CA1) |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        s = settings(row)
        match = s["match"] + (", normalised" if s["normalise"] else "")
        lines.append("| {} | {} | {} |".format(
            row["label"], match, " | ".join(cell(row, st) for st in ("vc", "pr", "pc", "pm"))))
    return "\n".join(lines)


def results_table(row):
    """Markdown table for one row, every stage, with the paper's numbers."""
    names = {"pixels": "Raw pixels", "vc": "VC(EC) alone", "pr": "PR(EC-CA3)",
             "pc": "PC(CA3)", "pm": "PM(CA1)"}
    lines = ["| Stage scored | This repo | Paper |", "|---|---|---|"]
    for stage in STAGES:
        paper = "{:.1%}".format(PAPER[stage]) if stage in PAPER else "-"
        lines.append("| {} | {} | {} |".format(names[stage], cell(row, stage), paper))
    return "\n".join(lines)


def write_csv(rows, path):
    fields = ["file", "label", "match", "normalise", "normalise_input", "loss",
              "interest_filter", "centre", "ca3_rule", "checkpoints", "seeds", "spread"]
    for stage in STAGES:
        fields += [stage + "_mean", stage + "_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = {"file": row["file"], "label": row["label"], **settings(row)}
            for stage in STAGES:
                s = row["summary"].get(stage, {})
                out[stage + "_mean"] = "{:.4f}".format(s["mean"]) if s else ""
                out[stage + "_std"] = "{:.4f}".format(s["std"]) if s else ""
            w.writerow(out)


def build(rows):
    by_file = {r["file"]: r for r in rows}
    parts = []
    if "oneshot.json" in by_file:
        parts += ["## Results, default settings", "", results_table(by_file["oneshot.json"]), ""]
    for group, title in (("grid", "Closing the gap"), ("ablation", "Ablations"),
                         ("device", "Devices")):
        chosen = [r for r in rows if r["group"] == group]
        if chosen:
            parts += ["## " + title, "", grid_table(chosen), ""]
    return "\n".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)

    rows = load_rows(args.results_dir)
    md = build(rows)
    print(md)
    Path(args.results_dir, "tables.md").write_text(md)
    write_csv(rows, Path(args.results_dir, "grid.csv"))
    print("Wrote {}/tables.md and {}/grid.csv".format(args.results_dir, args.results_dir))


if __name__ == "__main__":
    main()

"""
Pick the model train.py uses for one species, from modelBakeoff.py's outputs

Reads hyperParams/[species]-scores.csv and hyperParams/[species].json and prints
the ranked scores, the chosen bakeoff row and the line to put in train_model().

Rule: the lowest test RMSE wins, unless a simpler model is within NEAR_TIE of
it, in which case the simplest of those wins.

Usage, from a location folder:
    ../.venv/bin/python ../.claude/skills/run-pipeline/scripts/pick_model.py ../.data/<Location>/hyperParams <species>
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Models within this fraction of the lowest RMSE count as tied with it. Test
# splits are only ~200 hours, so smaller gaps are mostly noise.
NEAR_TIE = 0.02

# Lower is simpler. Linear models also extrapolate past the colocation range,
# which KNN, DT, RF and XGB can't, and current data comes from a different place and season.
SIMPLICITY = {"Linear": 0, "Ridge": 1, "Lasso": 1, "KNN": 2, "DT": 3, "RF": 4, "XGB": 5}

# Class for each bakeoff model name, with the constructor args the bakeoff
# gave it on top of the tuned params
CONSTRUCTORS = {
    "DT": ("DecisionTreeRegressor", {"random_state": 42}),
    "Linear": ("LinearRegression", {}),
    "Ridge": ("Ridge", {"random_state": 42}),
    "Lasso": ("Lasso", {"random_state": 42}),
    "KNN": ("KNeighborsRegressor", {}),
    "RF": ("RandomForestRegressor", {"random_state": 42}),
    "XGB": ("XGBRegressor", {"random_state": 42}),
}

# On an exact tie, Grid comes first so the pick doesn't depend on row order
SEARCH_ORDER = {"Grid": 0, "Random": 1}

METRICS = ("MSE", "RMSE", "MAE", "R2", "Time")


def read_scores(path: Path):
    """
    Rows of the scores csv, with the row name split into model and search

    Returns
    -------
    rows: list of dict
        Keys name ("KNN (Grid)"), model ("KNN"), search ("Grid") and each metric
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            model, search = r["Model"].rsplit(" (", 1)
            row = {"name": r["Model"], "model": model, "search": search.rstrip(")")}
            # pandas writes NaN as an empty cell
            row.update({m: float(r[m]) if r[m] else math.nan for m in METRICS})
            rows.append(row)
    return rows


def choose(rows):
    """
    Apply the selection rule

    Returns
    -------
    best: dict
        Row with the lowest RMSE
    chosen: dict
        Row train.py should use
    """
    unknown = sorted({r["model"] for r in rows} - SIMPLICITY.keys())
    if unknown:
        sys.exit(f"No selection rule for {', '.join(unknown)}: add them to SIMPLICITY and CONSTRUCTORS")

    scored = [r for r in rows if not math.isnan(r["RMSE"])]
    best = min(scored, key=lambda r: r["RMSE"])
    tied = [r for r in scored if r["RMSE"] <= best["RMSE"] * (1 + NEAR_TIE)]
    chosen = min(tied, key=lambda r: (SIMPLICITY[r["model"]], r["RMSE"], SEARCH_ORDER[r["search"]]))
    return best, chosen


def constructor_args(model: str, params: dict, quote: str = "'") -> tuple:
    """
    Class name and keyword arguments that build the model

    Parameters
    ----------
    model : str
        Bakeoff model name, e.g. "Ridge"
    params : dict
        Tuned params from the bakeoff JSON
    quote : str
        Quote character for string values

    Returns
    -------
    cls: str
        e.g. "Ridge"
    args: str
        e.g. "alpha=0.001, random_state=42, solver='svd'"
    """
    cls, extra = CONSTRUCTORS[model]
    merged = {**params, **extra}
    fmt = lambda v: f"{quote}{v}{quote}" if isinstance(v, str) else repr(v)
    return cls, ", ".join(f"{k}={fmt(v)}" for k, v in sorted(merged.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyperParams", type=Path, help="the location's .data/<Location>/hyperParams folder")
    parser.add_argument("species")
    args = parser.parse_args()

    scoresPath = args.hyperParams / f"{args.species}-scores.csv"
    paramsPath = args.hyperParams / f"{args.species}.json"
    for path in (scoresPath, paramsPath):
        if not path.exists():
            sys.exit(f"{path} not found: run modelBakeoff.py {args.species} first")

    rows = read_scores(scoresPath)
    best, chosen = choose(rows)
    params = json.loads(paramsPath.read_text())[chosen["model"]][chosen["search"]]

    print(f"Bakeoff scores for {args.species}, lowest RMSE first:")
    print(f"  {'Model':<16} {'RMSE':>10} {'R2':>8} {'MAE':>10}")
    for r in sorted(rows, key=lambda r: (math.isnan(r["RMSE"]), r["RMSE"])):
        print(f"  {r['name']:<16} {r['RMSE']:>10.4f} {r['R2']:>8.4f} {r['MAE']:>10.4f}")
    print()

    print(f"Lowest RMSE: {best['name']} ({best['RMSE']:.4f})")
    if chosen is best:
        reason = "lowest RMSE"
    else:
        gap = 100 * (chosen["RMSE"] / best["RMSE"] - 1)
        reason = f"{gap:.1f}% above the lowest RMSE, within {100 * NEAR_TIE:g}%, and simpler than {best['model']}"
    print(f"Chosen: {chosen['name']} ({reason})")
    print(f"  RMSE {chosen['RMSE']:.4f}  R2 {chosen['R2']:.4f}  MAE {chosen['MAE']:.4f}")
    if chosen["R2"] < 0:
        print("  WARNING: R2 is below 0, so this model predicts worse than the test-split mean")
    print()

    cls, codeArgs = constructor_args(chosen["model"], params)
    print("Line for this species' branch of train_model():")
    print(f"    model = {cls}({codeArgs})")
    print()

    # docs/featureselection.md writes string values in double quotes
    _, docArgs = constructor_args(chosen["model"], params, quote='"')
    print("Row for the location's models table in docs/featureselection.md:")
    print(f"| {args.species} | `{cls}` | `{docArgs}` |")
    print()
    print(f"train.py should then report Test MSE {chosen['MSE']:.6g} and Test R2 {chosen['R2']:.6g}")


if __name__ == "__main__":
    main()

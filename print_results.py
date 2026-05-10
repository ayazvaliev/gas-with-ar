#!/usr/bin/env python3
"""Print a formatted FID results table from a JSON file produced by eval_checkpoints.py.

Results are grouped by NFE; within each group rows are sorted by FID descending
(worst → best top-to-bottom) so configs are easy to compare per step count.

Usage:
  python print_results.py results.json
  python print_results.py results.json --comet_project my-project
  python print_results.py results.json --comet_project my-project --comet_workspace myworkspace
"""
from __future__ import annotations

import json

import click


def _print_table(results: list[dict]) -> None:
    run_w  = max(len(r["run_name"]) for r in results)
    run_w  = max(run_w, len("Run name"))
    iter_w = max(len(str(r["ckpt_iter"])) for r in results)
    iter_w = max(iter_w, len("Iter"))

    total_w = run_w + iter_w + 5 + 5 + 10 + 12
    sep     = "-" * total_w
    hdr     = f"{'Run name':<{run_w}}  {'Iter':>{iter_w}}  {'NFE':>5}  {'FID':>10}"

    print(f"\n{'='*total_w}")
    print("EVALUATION RESULTS")
    print("=" * total_w)

    for nfe in sorted(set(r["nfe"] for r in results)):
        group = [r for r in results if r["nfe"] == nfe]
        # None FIDs sort last; otherwise descending (worst first)
        group.sort(key=lambda r: (r["fid"] if r["fid"] is not None else float("inf")))

        print(f"\n  NFE = {nfe}")
        print(hdr)
        print(sep)
        for r in group:
            fid_s = f"{r['fid']:.4f}" if r["fid"] is not None else "N/A"
            print(f"{r['run_name']:<{run_w}}  {r['ckpt_iter']:>{iter_w}}  {r['nfe']:>5}  {fid_s:>10}")

    print("\n" + "=" * total_w)


def _log_to_comet(results: list[dict], project: str, workspace: str | None) -> None:
    try:
        import comet_ml
    except ImportError:
        print("[ERROR] comet_ml is not installed. Run: pip install comet-ml")
        return

    from datetime import datetime

    kwargs = dict(project_name=project)
    if workspace:
        kwargs["workspace"] = workspace

    experiment = comet_ml.Experiment(**kwargs)
    experiment.set_name(datetime.now().strftime("fid_eval_%Y-%m-%d_%H-%M-%S"))

    # One table per NFE group, sorted descending by FID (worst → best)
    for nfe in sorted(set(r["nfe"] for r in results)):
        group = [r for r in results if r["nfe"] == nfe]
        group.sort(key=lambda r: (r["fid"] if r["fid"] is not None else float("inf")))
        experiment.log_table(
            f"fid_nfe{nfe}.csv",
            tabular_data=[[r["run_name"], r["ckpt_iter"], r["nfe"], r["fid"]] for r in group],
            headers=["run_name", "ckpt_iter", "nfe", "fid"],
        )

    # Per-NFE FID metrics so the CometML chart view shows one curve per run
    for r in results:
        if r["fid"] is None:
            continue
        experiment.log_metric(
            f"fid_nfe{r['nfe']}",
            r["fid"],
            step=int(r["ckpt_iter"]),
            epoch=None,
        )
        experiment.log_parameter(f"run_{r['run_name']}_nfe{r['nfe']}", r["fid"])

    experiment.end()
    print(f"\n[comet] Results logged to project '{project}'."
          f"  URL: {experiment.url}")


@click.command()
@click.argument("results_json", type=click.Path(exists=True))
@click.option("--comet_project", default=None,
              help="CometML project name. When set, exports the results table to CometML.")
@click.option("--comet_workspace", default=None,
              help="CometML workspace (defaults to your personal workspace).")
def main(results_json: str, comet_project: str | None, comet_workspace: str | None) -> None:
    with open(results_json) as f:
        results: list[dict] = json.load(f)

    if not results:
        print("No results found in file.")
        return

    _print_table(results)

    if comet_project:
        _log_to_comet(results, comet_project, comet_workspace)


if __name__ == "__main__":
    main()

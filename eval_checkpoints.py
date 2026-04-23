#!/usr/bin/env python3
from __future__ import annotations
"""
Batch evaluation: scan checkpoints/, match each run to its config, generate
images, compute FID, and print a summary table.

Checkpoint dirs are expected to follow the naming convention produced by
training.py:  <run_name>_MM_DD_HH_MM_SS
The latest .pt file (by iteration number) in each dir is used.

Usage examples:
  python eval_checkpoints.py --fid_ref fid-refs/edm/cifar10-32x32.npz
  python eval_checkpoints.py \\
      --checkpoints_dir checkpoints --configs_dir configs \\
      --fid_ref fid-refs/edm/cifar10-32x32.npz \\
      --seeds 50000-99999 --batch 1024
"""

import os
import re
import sys
import subprocess
from pathlib import Path

import click
import yaml
from ml_collections import ConfigDict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_streaming(cmd: list) -> tuple[str, int]:
    """Run *cmd*, stream every line to stdout in real-time, return (output, rc)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    return "".join(lines), proc.returncode


def _extract_run_name(dirname: str) -> str:
    """Strip the trailing _MM_DD_HH_MM_SS date suffix from a checkpoint dir name."""
    m = re.match(r"^(.+)_(\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$", dirname)
    return m.group(1) if m else dirname


def _find_last_checkpoint(ckpt_dir: str) -> str | None:
    """Return path of the highest-numbered .pt file in ckpt_dir, or None."""
    pts = []
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".pt"):
            try:
                pts.append((int(fname[:-3]), fname))
            except ValueError:
                pass
    if not pts:
        return None
    pts.sort()
    return os.path.join(ckpt_dir, pts[-1][1])


def _parse_nfe_list(solver_config) -> list[int] | None:
    """Return NFE list from solver_config, or None if it cannot be determined."""
    steps_ratios = getattr(solver_config, "steps_ratios", None)
    if steps_ratios is not None:
        return sorted(int(k) for k in steps_ratios.keys())
    if solver_config.steps is not None:
        return [int(solver_config.steps)]
    return None


def _parse_fid(output: str) -> float | None:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("FID:"):
            try:
                return float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--checkpoints_dir", default="checkpoints", show_default=True,
              help="Root directory that contains per-run checkpoint subdirs.")
@click.option("--configs_dir", default="configs", show_default=True,
              help="Root directory to search (recursively) for YAML configs.")
@click.option("--fid_ref", required=True, type=str,
              help="Path to .npz FID reference statistics file.")
@click.option("--outdir", default="data/eval", show_default=True,
              help="Root output directory for generated images.")
@click.option("--seeds", default="50000-99999", show_default=True,
              help="Seed range passed to generate.py (e.g. 50000-99999).")
@click.option("--batch", default=1024, show_default=True,
              help="Batch size for generate.py and FID computation.")
@click.option("--steps", "steps_override", default=None, type=str,
              help="Comma-separated NFE override applied to ALL runs. "
                   "Required when a config has steps=null and no steps_ratios.")
@click.option("--n_gpu", default=1, show_default=True,
              help="Number of GPUs for torchrun (fid.py).")
@click.option("--skip_generate", is_flag=True, default=False,
              help="Skip image generation if output dir already exists.")
def main(
    checkpoints_dir, configs_dir, fid_ref, outdir,
    seeds, batch, steps_override, n_gpu, skip_generate,
):
    # ------------------------------------------------------------------
    # 1. Parse all configs → {run_name: (config_path, solver_config)}
    # ------------------------------------------------------------------
    run_name_to_cfg: dict[str, tuple[str, object]] = {}
    for root, _, files in os.walk(configs_dir):
        for fname in sorted(files):
            if not fname.endswith((".yaml", ".yml")):
                continue
            config_path = os.path.join(root, fname)
            try:
                with open(config_path) as f:
                    cfg = ConfigDict(yaml.safe_load(f))
                run_name = getattr(cfg.logging, "run_name", None)
                if run_name is None:
                    continue
                if run_name in run_name_to_cfg:
                    print(
                        f"[WARN] Duplicate run_name '{run_name}' in {config_path}; "
                        f"already mapped from {run_name_to_cfg[run_name][0]}. Keeping first."
                    )
                else:
                    run_name_to_cfg[run_name] = (config_path, cfg.student_solver_config)
            except Exception as exc:
                print(f"[WARN] Could not parse {config_path}: {exc}")

    print(f"Parsed {len(run_name_to_cfg)} configs with run_name set.")

    # ------------------------------------------------------------------
    # 2. Scan checkpoint dirs → [(run_name, ckpt_dir, ckpt_path)]
    # ------------------------------------------------------------------
    if not os.path.isdir(checkpoints_dir):
        raise click.UsageError(f"Checkpoints directory not found: {checkpoints_dir}")

    ckpt_entries = []
    for dname in sorted(os.listdir(checkpoints_dir)):
        full = os.path.join(checkpoints_dir, dname)
        if not os.path.isdir(full):
            continue
        run_name = _extract_run_name(dname)
        ckpt_path = _find_last_checkpoint(full)
        if ckpt_path is None:
            print(f"[WARN] No .pt file in {full}, skipping.")
            continue
        ckpt_entries.append((run_name, full, ckpt_path))

    print(f"Found {len(ckpt_entries)} checkpoint dirs with .pt files.")

    # ------------------------------------------------------------------
    # 3. Match checkpoints → configs
    # ------------------------------------------------------------------
    matched = []
    for run_name, ckpt_dir, ckpt_path in ckpt_entries:
        if run_name in run_name_to_cfg:
            config_path, solver_cfg = run_name_to_cfg[run_name]
            matched.append((run_name, ckpt_dir, ckpt_path, config_path, solver_cfg))
        else:
            print(
                f"[WARN] No config found for checkpoint dir '{os.path.basename(ckpt_dir)}' "
                f"(run_name='{run_name}'). Skipping."
            )

    print(f"Matched {len(matched)} checkpoint(s) to configs.\n")
    if not matched:
        print("Nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # 4. Generate images + compute FID/LPIPS
    # ------------------------------------------------------------------
    results = []

    for run_name, ckpt_dir, ckpt_path, config_path, solver_cfg in matched:
        ckpt_iter = Path(ckpt_path).stem          # e.g. "5000"
        exp_outdir = os.path.join(outdir, run_name, ckpt_iter)

        print(f"\n{'='*72}")
        print(f"  Run      : {run_name}")
        print(f"  Ckpt     : {ckpt_path}  (iter {ckpt_iter})")
        print(f"  Config   : {config_path}")
        print(f"  Out dir  : {exp_outdir}")
        print(f"{'='*72}")

        # Determine NFE list
        if steps_override is not None:
            nfe_list = [int(s.strip()) for s in steps_override.split(",")]
        else:
            nfe_list = _parse_nfe_list(solver_cfg)
            if nfe_list is None:
                print(
                    f"[WARN] Cannot determine NFE list for '{run_name}' "
                    f"(steps=null, no steps_ratios). Pass --steps to override. Skipping."
                )
                continue

        is_multi_nfe = (
            getattr(solver_cfg, "t_parametrization", None) == "ar_model"
            and len(nfe_list) > 1
        )
        steps_str = ",".join(str(n) for n in nfe_list)

        # ---- generate ------------------------------------------------
        images_exist = (
            skip_generate
            and os.path.isdir(os.path.join(exp_outdir, "images"))
        )
        if images_exist:
            print(f"[skip] Images already at {exp_outdir}/images — skipping generation.")
        else:
            gen_cmd = [
                sys.executable, "generate.py",
                f"--config={config_path}",
                f"--outdir={exp_outdir}",
                f"--seeds={seeds}",
                f"--batch={batch}",
                f"--steps={steps_str}",
                f"--checkpoint_path={ckpt_path}",
            ]
            print(f"\n[generate] {' '.join(gen_cmd)}")
            _, rc = _run_streaming(gen_cmd)
            if rc != 0:
                print(f"[ERROR] generate.py exited with code {rc}. Skipping FID for '{run_name}'.")
                continue

        # ---- FID per NFE --------------------------------------------
        for nfe in nfe_list:
            images_dir = (
                os.path.join(exp_outdir, "images", str(nfe))
                if is_multi_nfe
                else os.path.join(exp_outdir, "images")
            )

            if not os.path.isdir(images_dir):
                print(f"[WARN] Images directory not found: {images_dir}")
                continue

            fid_cmd = [
                "torchrun", "--standalone", f"--nproc_per_node={n_gpu}",
                "fid.py", "calc",
                f"--images={images_dir}",
                f"--ref={fid_ref}",
                f"--batch={batch}",
            ]

            print(f"\n[fid] NFE={nfe}  {' '.join(fid_cmd)}")
            output, rc = _run_streaming(fid_cmd)

            fid_val = _parse_fid(output)
            if fid_val is None:
                print(f"[WARN] Could not parse FID from fid.py output (exit {rc}).")

            results.append(dict(
                run_name=run_name,
                ckpt_dir=os.path.basename(ckpt_dir),
                ckpt_iter=ckpt_iter,
                nfe=nfe,
                fid=fid_val,
            ))
            print(f"  → NFE={nfe}  FID={fid_val}")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    if not results:
        print("\nNo results collected.")
        return

    run_w  = max(len(r["run_name"]) for r in results)
    run_w  = max(run_w, len("Run name"))
    iter_w = max(len(r["ckpt_iter"]) for r in results)
    iter_w = max(iter_w, len("Iter"))

    sep = "-" * (run_w + iter_w + 5 + 5 + 10 + 12)
    hdr = f"{'Run name':<{run_w}}  {'Iter':>{iter_w}}  {'NFE':>5}  {'FID':>10}"

    print(f"\n\n{'='*len(sep)}")
    print("EVALUATION RESULTS")
    print("=" * len(sep))
    print(hdr)
    print(sep)

    for r in results:
        fid_s = f"{r['fid']:.4f}" if r["fid"] is not None else "N/A"
        print(f"{r['run_name']:<{run_w}}  {r['ckpt_iter']:>{iter_w}}  {r['nfe']:>5}  {fid_s:>10}")

    print("=" * len(sep))


if __name__ == "__main__":
    main()

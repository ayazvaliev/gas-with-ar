"""
Convert AR model checkpoint from the old nn.MultiheadAttention layout to the
new explicit qkv_proj layout.

Key renames (all other parameters are identical and require no conversion):
  decoder.self_attn.in_proj_weight  →  decoder.qkv_proj.weight
  decoder.self_attn.in_proj_bias    →  decoder.qkv_proj.bias
  decoder.self_attn.out_proj.weight →  decoder.out_proj.weight
  decoder.self_attn.out_proj.bias   →  decoder.out_proj.bias

nn.MultiheadAttention stores in_proj_weight as [3*d, d] with Q rows first,
then K, then V — matching the chunk(3, dim=-1) order in the new forward.

Usage (single checkpoint):
    python convert_ar_checkpoint.py old.pt new.pt

Usage (batch — whole checkpoints root dir):
    python convert_ar_checkpoint.py --dir checkpoints/
    python convert_ar_checkpoint.py --dir checkpoints/ --subdir converted
"""

from __future__ import annotations
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch


_RENAMES = {
    "self_attn.in_proj_weight":  "qkv_proj.weight",
    "self_attn.in_proj_bias":    "qkv_proj.bias",
    "self_attn.out_proj.weight": "out_proj.weight",
    "self_attn.out_proj.bias":   "out_proj.bias",
}


def _convert_named_state_dict(sd: dict) -> tuple[dict, int]:
    """Rename self_attn keys → qkv/out_proj keys. Returns (new_sd, n_renamed)."""
    new_sd = OrderedDict()
    n_renamed = 0
    for key, val in sd.items():
        new_key = key
        for old_suffix, new_suffix in _RENAMES.items():
            if key.endswith(old_suffix):
                # Preserve the prefix (e.g. "ar_model.decoder.")
                new_key = key[: -len(old_suffix)] + new_suffix
                n_renamed += 1
                break
        new_sd[new_key] = val
    return new_sd, n_renamed


def _convert_ema(ema_sd: dict, old_names: list[str], new_names: list[str]) -> dict:
    """
    Rebuild shadow_params for the new parameter ordering.

    EMA stores shadow_params as a positional list mirroring gs_wrapper.parameters().
    Since we only rename keys (no merging/splitting), the list length is unchanged;
    we just need to reorder to match the new parameter order.
    """
    shadow = list(ema_sd["shadow_params"])
    if len(shadow) != len(old_names):
        raise ValueError(
            f"EMA shadow_params length ({len(shadow)}) != "
            f"model state dict length ({len(old_names)})"
        )

    old_name_to_shadow = dict(zip(old_names, shadow))

    # Build a name→name mapping for the renamed keys
    old_to_new = {}
    for old in old_names:
        new = old
        for old_suffix, new_suffix in _RENAMES.items():
            if old.endswith(old_suffix):
                new = old[: -len(old_suffix)] + new_suffix
                break
        old_to_new[old] = new

    new_to_shadow = {old_to_new[k]: v for k, v in old_name_to_shadow.items()}

    new_shadow = [new_to_shadow[name] for name in new_names]
    return {**ema_sd, "shadow_params": new_shadow}


def convert_one(input_path: Path, output_path: Path) -> None:
    print(f"Loading {input_path} …")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    new_ckpt = dict(ckpt)

    # ── model state dict ──────────────────────────────────────────────────────
    model_val = ckpt.get("model")
    old_param_names = new_param_names = None

    if isinstance(model_val, list):
        # Checkpoint was saved via gs_wrapper.parameters() — a flat positional list.
        # The old (nn.MultiheadAttention) and new (explicit qkv_proj) architectures register
        # their attention parameters in the same order, so the list is already compatible
        # with the new arch.  No renaming or reordering needed for either model or EMA.
        print(
            f"  model state dict : 'model' is a positional parameter list "
            f"({len(model_val)} tensors) — no key renaming needed, EMA order is unchanged"
        )
    elif isinstance(model_val, (dict, OrderedDict)) and all(isinstance(k, str) for k in model_val):
        new_sd, n = _convert_named_state_dict(model_val)
        if n:
            old_param_names = list(model_val.keys())
            new_param_names = list(new_sd.keys())
            new_ckpt["model"] = new_sd
            print(f"  model state dict : {n} key(s) renamed")
        else:
            print("  model state dict : no self_attn keys found — already converted or different layout")
    else:
        print(
            f"  model state dict : 'model' is {type(model_val).__name__}, not a named dict — skipped\n"
            "                     (EMA conversion also skipped)"
        )

    # ── EMA shadow_params ─────────────────────────────────────────────────────
    ema_val = ckpt.get("ema")
    if ema_val is not None and "shadow_params" in ema_val:
        if old_param_names is None:
            # Covers both the list case (no reorder needed) and the already-converted case.
            print(f"  EMA shadow_params : {len(ema_val['shadow_params'])} tensors — no reorder needed")
        else:
            new_ema = _convert_ema(ema_val, old_param_names, new_param_names)
            new_ckpt["ema"] = new_ema
            print(f"  EMA shadow_params : {len(new_ema['shadow_params'])} tensors remapped")
    else:
        print("  EMA shadow_params : not present — skipped")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path} …")
    torch.save(new_ckpt, output_path)
    print("Done.")


def convert_dir(root: Path, subdir: str, overwrite: bool = False) -> None:
    """
    Walk root/<exp_name>/*.pt and write converted files to root/<exp_name>/<subdir>/*.pt.
    Skips checkpoints that live inside a subdir (to avoid re-converting already-converted files).
    """
    pt_files = [
        p for p in root.rglob("*.pt")
        if p.parent.parent == root  # only one level deep: root/exp_name/*.pt
    ]

    if not pt_files:
        print(f"No .pt files found directly inside experiment folders under {root}")
        return

    for src in sorted(pt_files):
        dst = src.parent / subdir / src.name
        if dst.exists() and not overwrite:
            print(f"Skipping {src} (output already exists at {dst})")
            continue
        convert_one(src, dst)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Legacy positional args (single-file mode)
    parser.add_argument("input",  type=Path, nargs="?", help="Old checkpoint .pt file")
    parser.add_argument("output", type=Path, nargs="?", help="Output path for converted checkpoint")

    # Batch mode
    parser.add_argument(
        "--dir", type=Path, metavar="ROOT",
        help="Root checkpoints directory (ROOT/EXP_NAME/*.pt layout)",
    )
    parser.add_argument(
        "--subdir", default="converted", metavar="NAME",
        help="Subdirectory inside each experiment folder to write converted files (default: converted)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite already-converted checkpoints instead of skipping them",
    )

    args = parser.parse_args()

    if args.dir is not None:
        if args.input is not None or args.output is not None:
            sys.exit("Error: --dir cannot be combined with positional input/output arguments")
        if not args.dir.is_dir():
            sys.exit(f"Error: {args.dir} is not a directory")
        convert_dir(args.dir, args.subdir, overwrite=args.overwrite)
    else:
        if args.input is None or args.output is None:
            parser.error("Provide either --dir ROOT or both positional arguments: input output")
        if not args.input.exists():
            sys.exit(f"Error: {args.input} not found")
        convert_one(args.input, args.output)


if __name__ == "__main__":
    main()

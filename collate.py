import os
import pickle
from collections import defaultdict
from typing import Optional

import click
import torch
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _collate_dir(step_dir: str, max_samples: Optional[int]) -> dict:
    """Load and concatenate all .pt files from a directory.

    Args:
        step_dir: Directory containing ``<seed>.pt`` files.
        max_samples: If not None, stop once this many samples have been loaded.

    Returns:
        Dict with tensor/list values, trimmed to max_samples.
    """
    paths = sorted(p for p in os.listdir(step_dir) if p.endswith(".pt"))
    data = defaultdict(list)
    total = 0

    pbar = tqdm(total=max_samples or float("inf"), desc=os.path.basename(step_dir), leave=False)
    for p in paths:
        checkpoint = torch.load(os.path.join(step_dir, p), weights_only=False)
        diff = None
        for k, v in checkpoint.items():
            if isinstance(v, torch.Tensor):
                data[k].append(v)
                diff = len(v)
            elif isinstance(v, list):
                data[k] += v
                diff = len(v)
            else:
                raise NotImplementedError(f"Unknown type {type(v)} to collate.")
        if diff is not None:
            total += diff
            pbar.update(diff)
        if max_samples is not None and total >= max_samples:
            break
    pbar.close()

    result = {}
    for k, v in data.items():
        combined = torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        result[k] = combined[:max_samples] if max_samples is not None else combined

    return result


def _concat_nfe_data(data_by_nfe: dict) -> dict:
    """Concatenate per-NFE dicts in sorted NFE order (all of NFE=4, then NFE=5, …)."""
    combined = defaultdict(list)
    for nfe in sorted(data_by_nfe.keys()):
        nfe_data = data_by_nfe[nfe]
        for k, v in nfe_data.items():
            if isinstance(v, torch.Tensor):
                combined[k].append(v)
            else:
                combined[k] += list(v)

    result = {}
    for k, v in combined.items():
        result[k] = torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
    return result


def _interleave_nfe_data(data_by_nfe: dict) -> dict:
    """Interleave per-NFE dicts so consecutive samples alternate NFEs.

    For NFE groups of equal length L and N groups, produces a dataset of
    N*L samples arranged as:
        [nfe_0[0], nfe_1[0], …, nfe_N[0], nfe_0[1], nfe_1[1], …]

    This guarantees that any suffix of length k*N has exactly k samples
    per NFE, which is required for the validation set assertion.
    """
    nfes = sorted(data_by_nfe.keys())
    lengths = {n: len(data_by_nfe[n][next(iter(data_by_nfe[n]))]) for n in nfes}
    assert len(set(lengths.values())) == 1, (
        f"All test NFE groups must have equal length for interleaving; "
        f"got: {dict(lengths)}"
    )
    per_nfe = next(iter(lengths.values()))

    # Pick a reference key set (all groups should have the same keys)
    keys = list(data_by_nfe[nfes[0]].keys())
    result = {}

    for k in keys:
        first = data_by_nfe[nfes[0]][k]
        if isinstance(first, torch.Tensor):
            # Stack along new dim, then flatten: [per_nfe, n_nfes, ...] -> [per_nfe*n_nfes, ...]
            stacked = torch.stack([data_by_nfe[n][k] for n in nfes], dim=1)
            result[k] = stacked.reshape(-1, *first.shape[1:])
        else:
            interleaved = []
            for i in range(per_nfe):
                for n in nfes:
                    interleaved.append(data_by_nfe[n][k][i])
            result[k] = interleaved

    return result


def _append_data(base: dict, extra: dict) -> dict:
    """Append ``extra`` to ``base`` for each key (tensor cat or list extend)."""
    out = {}
    all_keys = set(base) | set(extra)
    for k in all_keys:
        b = base.get(k)
        e = extra.get(k)
        if b is None:
            out[k] = e
        elif e is None:
            out[k] = b
        elif isinstance(b, torch.Tensor):
            out[k] = torch.cat([b, e], dim=0)
        else:
            out[k] = list(b) + list(e)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


@click.command()
@click.option(
    "--synt_dir",
    help="Path to the teacher directory. "
         "Multi-step mode: contains per-NFE subdirs named by step count; "
         "optionally a 'test/' subdir with the same structure for the test split. "
         "Single-step (legacy): flat directory of .pt files.",
    metavar="PATH",
    type=str,
    required=True,
)
@click.option(
    "--out_pkl", help="Path to output pkl dataset", metavar="PATH", type=str, required=True
)
@click.option(
    "--num_samples",
    help="Maximum number of train samples per NFE group (multi-step) or total "
         "(single-step). All available test samples are always included.",
    type=int,
    required=True,
    default=50000,
)
@click.option(
    "--steps",
    help="Comma-separated list of step counts to collate (e.g. '4,5,6,7'). "
         "If omitted, auto-detects digit-named subdirectories in synt_dir.",
    type=str,
    default=None,
)
def main(synt_dir, out_pkl, num_samples, steps):
    """Collate teacher data into a single pkl for training.

    \b
    Single-step (legacy):
      python collate.py --synt_dir=flat_dir --out_pkl=out.pkl --num_samples=2400

    \b
    Multi-step:
      python collate.py --synt_dir=dataset_root --out_pkl=out.pkl \\
                        --num_samples=600 --steps=4,5,6,7
      Expects: dataset_root/4/, dataset_root/5/, …
      Optionally: dataset_root/test/4/, dataset_root/test/5/, …
      Test samples (if present) are interleaved and appended to the END of the
      pkl so that val_indices = range(len-val_size, len) has equal NFE ratios.
    """
    assert os.path.splitext(out_pkl)[1] == ".pkl"

    # Detect whether multi-step
    if steps is not None:
        steps_list = sorted(int(s.strip()) for s in steps.split(","))
    else:
        subdirs = [
            d for d in os.listdir(synt_dir)
            if os.path.isdir(os.path.join(synt_dir, d)) and d.isdigit()
        ]
        if subdirs:
            steps_list = sorted(int(d) for d in subdirs)
            print(f"Auto-detected step subdirectories: {steps_list}")
        else:
            steps_list = None  # single-step legacy

    # ------------------------------------------------------------------ #
    # Multi-step                                                           #
    # ------------------------------------------------------------------ #
    if steps_list is not None:
        # --- Train data ---
        train_by_nfe = {}
        for nfe in steps_list:
            step_dir = os.path.join(synt_dir, str(nfe))
            assert os.path.isdir(step_dir), f"Missing train dir: {step_dir}"
            print(f"Collating train nfe={nfe} from {step_dir}")
            train_by_nfe[nfe] = _collate_dir(step_dir, num_samples)
            actual = len(train_by_nfe[nfe][next(iter(train_by_nfe[nfe]))])
            print(f"  -> {actual} samples")

        combined = _concat_nfe_data(train_by_nfe)

        # --- Test data (optional) ---
        test_dir = os.path.join(synt_dir, "test")
        if os.path.isdir(test_dir):
            test_by_nfe = {}
            for nfe in steps_list:
                nfe_test_dir = os.path.join(test_dir, str(nfe))
                if not os.path.isdir(nfe_test_dir):
                    print(f"Warning: test dir not found for nfe={nfe}, skipping test split")
                    test_by_nfe = {}
                    break
                print(f"Collating test  nfe={nfe} from {nfe_test_dir}")
                test_by_nfe[nfe] = _collate_dir(nfe_test_dir, None)  # take all
                actual = len(test_by_nfe[nfe][next(iter(test_by_nfe[nfe]))])
                print(f"  -> {actual} samples")

            if test_by_nfe:
                interleaved_test = _interleave_nfe_data(test_by_nfe)
                combined = _append_data(combined, interleaved_test)
                n_test = len(interleaved_test[next(iter(interleaved_test))])
                print(f"Appended {n_test} interleaved test samples at the end.")

        total = len(combined[next(iter(combined))])
        print(f"Total dataset size: {total}")

    # ------------------------------------------------------------------ #
    # Single-step legacy                                                   #
    # ------------------------------------------------------------------ #
    else:
        paths = sorted(os.listdir(synt_dir))
        data = defaultdict(list)
        total_loaded = 0
        pbar = tqdm(total=num_samples)

        for p in paths:
            checkpoint = torch.load(os.path.join(synt_dir, p), weights_only=False)
            diff = None
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor):
                    data[k].append(v)
                    diff = len(v)
                elif isinstance(v, list):
                    data[k] += v
                    diff = len(v)
                else:
                    raise NotImplementedError(f"Unknown {type(v)} type to collate.")
            if diff is not None:
                total_loaded += diff
                pbar.update(diff)
            if total_loaded >= num_samples:
                break
        pbar.close()

        combined = {}
        for k, v in data.items():
            combined[k] = torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
            combined[k] = combined[k][:num_samples]
            assert len(combined[k]) == num_samples, (
                f"Data shape is {len(combined[k])}, expected {num_samples}"
            )

    with open(out_pkl, "wb") as f:
        pickle.dump(combined, f)

    print(f"Saved dataset to {out_pkl}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

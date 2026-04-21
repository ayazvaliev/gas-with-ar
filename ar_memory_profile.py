#!/usr/bin/env python3
"""
Profile AR model parameter count and peak CUDA memory for inference and training.

Produces per-NFE rows for:
  - AR model activations only (isolated forward / forward+backward)
  - AR model weights + activations
  - Full pipeline peak (solver + diffusion backbone + LPIPS)

Usage:
    python ar_memory_profile.py --config configs/edm/cifar10_ar_mixed.yaml
    python ar_memory_profile.py --config configs/edm/cifar10_ar.yaml --steps 5,6,8 --n_iters 10
"""

import statistics

import click
import torch
import yaml
from ml_collections import ConfigDict

from src.gas.models import get_gs_wrapper, load_base_model
from src.gas.synt_data import SyntDataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mib(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.3f} MiB"


def _measure_peak(fn, n_iters: int, device: torch.device) -> int:
    """Median peak CUDA memory delta (bytes) over n_iters calls to fn().

    Delta is measured relative to the allocation present before the first
    call so that model weights already on GPU are excluded.
    """
    peaks = []
    for _ in range(n_iters):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        fn()
        torch.cuda.synchronize(device)
        peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
    return int(statistics.median(peaks))


def _zero_grads(params):
    for p in params:
        p.grad = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", required=True, type=str,
              help="Path to YAML config (same format as main.py).")
@click.option("--steps", "steps_str", default=None, type=str,
              help="Comma-separated NFE values to profile. "
                   "Defaults to steps_ratios keys or config.steps.")
@click.option("--batch_size", default=8, show_default=True,
              help="Batch size used for every measurement.")
@click.option("--n_iters", default=5, show_default=True,
              help="Forward passes per estimate; median is reported.")
def main(config_path: str, steps_str: str, batch_size: int, n_iters: int) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for memory profiling.")

    device = torch.device("cuda")

    with open(config_path) as f:
        config = ConfigDict(yaml.safe_load(f))

    solver_config = config.student_solver_config
    dataset_config = config.dataset

    assert solver_config.t_parametrization == "ar_model", (
        "This script only supports t_parametrization=ar_model configs."
    )

    # ------------------------------------------------------------------
    # Determine NFE list
    # ------------------------------------------------------------------
    if steps_str is not None:
        nfe_list = sorted(int(s.strip()) for s in steps_str.split(","))
    elif getattr(solver_config, "steps_ratios", None) is not None:
        nfe_list = sorted(int(k) for k in solver_config.steps_ratios.keys())
    elif solver_config.steps is not None:
        nfe_list = [int(solver_config.steps)]
    else:
        raise click.UsageError(
            "Cannot determine NFE list. Use --steps or set steps/steps_ratios in config."
        )

    max_nfe = max(nfe_list)
    print(f"NFE list  : {nfe_list}")
    print(f"batch_size: {batch_size}    n_iters: {n_iters}")

    # ------------------------------------------------------------------
    # Prepare solver config (mirrors main.py / generate.py logic)
    # ------------------------------------------------------------------
    solver_config.loss_config.loss_type = "GS"
    solver_config.steps = max_nfe
    ar_learn_correctors = (
        getattr(solver_config, "learn_correctors", False)
        and solver_config.t_parametrization == "ar_model"
    )
    if not ar_learn_correctors and solver_config.order is None:
        solver_config.order = max_nfe

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading base model from {config.model.path} …")
    model_config = config.model
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale
    base_model = load_base_model(model_config, device)

    gs_wrapper = get_gs_wrapper(base_model, solver_config)
    ar_model = gs_wrapper.ar_model

    # ------------------------------------------------------------------
    # AR model parameter statistics
    # ------------------------------------------------------------------
    n_params = sum(p.numel() for p in ar_model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in ar_model.parameters())

    print()
    print("=" * 58)
    print(f"  AR model total parameters : {n_params:>14,}")
    print(f"  AR model weight memory    : {_mib(weight_bytes):>14}")
    print("=" * 58)

    image_shape = (
        batch_size,
        base_model.image_channels,
        base_model.image_size,
        base_model.image_size,
    )

    # ------------------------------------------------------------------
    # INFERENCE  (torch.no_grad)
    # ------------------------------------------------------------------
    print(f"\n── Inference  (no_grad, batch={batch_size}, n_iters={n_iters}) ──")
    print(f"{'NFE':>5}  {'AR act':>12}  {'AR w+act':>12}  {'Pipeline':>12}")
    print("-" * 48)

    gs_wrapper.eval()

    for nfe in nfe_list:
        # warm-up
        with torch.no_grad():
            ar_model(nfe - 1)
        noise = torch.randn(image_shape, device=device)
        with torch.no_grad():
            gs_wrapper.student_sampler_fn(noise, n_steps=nfe)

        def _ar_inf(n=nfe):
            with torch.no_grad():
                ar_model(n - 1)

        def _full_inf(n=nfe, _noise=noise):
            with torch.no_grad():
                gs_wrapper.student_sampler_fn(_noise, n_steps=n)

        ar_act   = _measure_peak(_ar_inf,   n_iters, device)
        full_act = _measure_peak(_full_inf,  n_iters, device)

        print(
            f"{nfe:>5}  {_mib(ar_act):>12}  "
            f"{_mib(ar_act + weight_bytes):>12}  "
            f"{_mib(full_act):>12}"
        )

    # ------------------------------------------------------------------
    # TRAINING  (forward + backward)
    # ------------------------------------------------------------------
    print(f"\n── Training  (forward+backward, batch={batch_size}, n_iters={n_iters}) ──")
    print("Loading dataset …")

    dataset = SyntDataset(dataset_path=dataset_config.teacher_pkl)
    # Wrap around with % in case batch_size > len(dataset)
    items = [dataset[i % len(dataset)] for i in range(batch_size)]

    noise_b  = torch.stack([it[0] for it in items]).to(device)
    images_b = torch.stack([it[1] for it in items]).to(device)
    latents_b = (
        torch.stack([it[2] for it in items]).to(device)
        if items[0][2] is not None else None
    )
    cond_b = (
        torch.stack([it[3] for it in items]).to(device)
        if isinstance(items[0][3], torch.Tensor)
        else [it[3] for it in items] if items[0][3] is not None
        else None
    )
    batch = (noise_b, images_b, latents_b, cond_b)

    print(f"{'NFE':>5}  {'AR act':>12}  {'AR w+act':>12}  {'Pipeline':>12}")
    print("-" * 48)

    gs_wrapper.train()
    ar_params = list(ar_model.parameters())
    gs_params  = list(gs_wrapper.parameters())

    for nfe in nfe_list:
        # warm-up
        ar_model(nfe - 1).sum().backward()
        _zero_grads(ar_params)

        def _ar_train(n=nfe):
            out = ar_model(n - 1)
            out.sum().backward()
            _zero_grads(ar_params)

        def _full_train(n=nfe, _batch=batch):
            res = gs_wrapper.forward(batch=_batch, n_steps_override=n)
            res["loss_total"].mean().backward()
            _zero_grads(gs_params)

        ar_act   = _measure_peak(_ar_train,   n_iters, device)
        full_act = _measure_peak(_full_train,  n_iters, device)

        print(
            f"{nfe:>5}  {_mib(ar_act):>12}  "
            f"{_mib(ar_act + weight_bytes):>12}  "
            f"{_mib(full_act):>12}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

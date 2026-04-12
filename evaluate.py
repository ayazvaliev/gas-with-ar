from collections import defaultdict
from typing import Optional

import torch

from src.gas.gs_wrapper import GSWrapper
from src.gas.synt_data import SyntDataLoaders
from src.gas.utils.loggers import (
    log_end_img,
    log_t_steps,
    log_t_steps_plot,
)
from comet_ml import Experiment

NOT_LOG_KEYS = ["timesteps", "x0_s", "x0_t", "latents_s"]


def _is_per_sample(v, batch_size: int) -> bool:
    """Return True if ``v`` is a per-sample tensor aligned with the batch."""
    return isinstance(v, torch.Tensor) and v.shape[0] == batch_size


@torch.no_grad()
def evaluate_wrapper(
    gs_wrapper: GSWrapper,
    data: SyntDataLoaders,
    device: torch.device,
    suff: str,
    global_step: int,
    experiment: Experiment,
) -> None:
    """Evaluate GS on the visualisation batch and test dataset.

    In mixed-NFE (AR) mode, separately logs:
    - A timestep plot per unique n_steps in vis_batch.
    - A student-vs-teacher image grid per unique n_steps in vis_batch.
    - Per-n_steps aggregate metrics computed on the full test dataset.
    """
    # ------------------------------------------------------------------ #
    # Visualisation batch                                                  #
    # ------------------------------------------------------------------ #
    vis_batch = [
        v.to(device) if isinstance(v, torch.Tensor) else v
        for v in data.vis_batch
    ]
    n_steps_vis: Optional[torch.Tensor] = (
        vis_batch[4] if len(vis_batch) == 5 and vis_batch[4] is not None else None
    )

    out_vis = gs_wrapper.forward(batch=vis_batch, return_timesteps=True, is_train=False)

    # Ensure pixel-space images are available
    if "x0_s" not in out_vis:
        with torch.no_grad():
            out_vis["x0_s"] = gs_wrapper.model.decode(out_vis["latents_s"])

    is_ar_mode = (
        n_steps_vis is not None
        and gs_wrapper.solver_config.t_parametrization == "ar_model"
    )

    if is_ar_mode:
        unique_steps = sorted(torch.unique(n_steps_vis).tolist())

        for nfe in [int(n) for n in unique_steps]:
            mask = (n_steps_vis == nfe).nonzero(as_tuple=True)[0]

            # Timestep trajectory for this NFE
            t_steps_nfe = gs_wrapper.get_timesteps_for_n(nfe)
            log_t_steps_plot(
                t_steps=t_steps_nfe,
                global_step=global_step,
                key=f"eval_image{suff}/t_steps_nfe{nfe}",
                experiment=experiment,
            )
            log_t_steps(
                t_steps=t_steps_nfe,
                global_step=global_step,
                key=f"t_stats{suff}/nfe{nfe}",
                experiment=experiment,
            )

            # Student vs teacher images for this NFE
            log_end_img(
                out_vis["x0_s"][mask],
                out_vis["x0_t"][mask],
                global_step=global_step,
                key=f"vis_stat{suff}/backward_end_inter_nfe{nfe}",
                experiment=experiment,
            )
    else:
        # Single-NFE: legacy behaviour
        log_t_steps_plot(
            t_steps=out_vis["timesteps"],
            global_step=global_step,
            key=f"eval_image{suff}/t_steps",
            experiment=experiment,
        )
        log_end_img(
            out_vis["x0_s"],
            out_vis["x0_t"],
            global_step=global_step,
            key=f"vis_stat{suff}/backward_end_inter",
            experiment=experiment,
        )

    # Aggregate vis metrics
    d_res = {}
    for k, v in out_vis.items():
        if k not in NOT_LOG_KEYS:
            d_res[f"vis_stat/{k}{suff}"] = v.mean().item()

    # ------------------------------------------------------------------ #
    # Test dataset                                                         #
    # ------------------------------------------------------------------ #
    log_d: dict = defaultdict(float)                       # aggregate
    per_nfe_d: dict = defaultdict(lambda: defaultdict(float))  # per-NFE
    per_nfe_count: dict = defaultdict(int)
    num_elements = 0

    out_test = None  # keep last batch for image logging
    n_steps_last: Optional[torch.Tensor] = None

    for batch in data.test_loader:
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        n_steps_batch: Optional[torch.Tensor] = (
            batch[4] if len(batch) == 5 and batch[4] is not None else None
        )

        out_test = gs_wrapper.forward(batch=batch, return_timesteps=False, is_train=False)
        bs = batch[0].shape[0]
        num_elements += bs
        n_steps_last = n_steps_batch

        for k, v in out_test.items():
            if k not in NOT_LOG_KEYS:
                log_d[k] += v.mean().item() * bs

        # Per-NFE accumulation
        if (
            n_steps_batch is not None
            and gs_wrapper.solver_config.t_parametrization == "ar_model"
        ):
            for nfe in torch.unique(n_steps_batch).tolist():
                nfe = int(nfe)
                mask = (n_steps_batch == nfe).nonzero(as_tuple=True)[0]
                count_n = len(mask)
                per_nfe_count[nfe] += count_n
                for k, v in out_test.items():
                    if k not in NOT_LOG_KEYS and _is_per_sample(v, bs):
                        per_nfe_d[nfe][k] += v[mask].mean().item() * count_n

    # Aggregate test metrics
    for k, v in log_d.items():
        d_res[f"val_stat/{k}{suff}"] = v / num_elements

    # Per-NFE test metrics
    for nfe, nfe_log in per_nfe_d.items():
        for k, v in nfe_log.items():
            d_res[f"val_stat_nfe{nfe}/{k}{suff}"] = v / per_nfe_count[nfe]

    experiment.log_metrics(d_res, step=global_step)

    # ------------------------------------------------------------------ #
    # Test batch image logging (last batch)                               #
    # ------------------------------------------------------------------ #
    if out_test is None:
        return

    if "x0_s" not in out_test:
        with torch.no_grad():
            out_test["x0_s"] = gs_wrapper.model.decode(out_test["latents_s"])

    if (
        n_steps_last is not None
        and gs_wrapper.solver_config.t_parametrization == "ar_model"
    ):
        for nfe in sorted(torch.unique(n_steps_last).tolist()):
            nfe = int(nfe)
            mask = (n_steps_last == nfe).nonzero(as_tuple=True)[0]
            log_end_img(
                out_test["x0_s"][mask],
                out_test["x0_t"][mask],
                global_step=global_step,
                key=f"val_stat{suff}/backward_end_inter_nfe{nfe}",
                experiment=experiment,
            )
    else:
        log_end_img(
            out_test["x0_s"],
            out_test["x0_t"],
            global_step=global_step,
            key=f"val_stat{suff}/backward_end_inter",
            experiment=experiment,
        )

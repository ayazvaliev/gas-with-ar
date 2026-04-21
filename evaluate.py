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

    In AR mode (t_parametrization='ar_model' with steps_ratios), separately logs
    timestep plots, image grids, and metrics for each NFE on the same test set.
    """
    steps_ratios = getattr(gs_wrapper.solver_config, 'steps_ratios', None)
    is_ar_mode = (
        steps_ratios is not None
        and gs_wrapper.solver_config.t_parametrization == "ar_model"
    )

    vis_batch = [
        v.to(device) if isinstance(v, torch.Tensor) else v
        for v in data.vis_batch
    ]

    d_res = {}

    if is_ar_mode:
        nfe_list = sorted(int(k) for k in steps_ratios.keys())

        # ------------------------------------------------------------------ #
        # Visualisation batch — one pass per NFE                              #
        # ------------------------------------------------------------------ #
        for nfe in nfe_list:
            out_vis = gs_wrapper.forward(
                batch=vis_batch, return_timesteps=True, is_train=False, n_steps_override=nfe
            )
            if "x0_s" not in out_vis:
                out_vis["x0_s"] = gs_wrapper.model.decode(out_vis["latents_s"])

            log_t_steps_plot(
                t_steps=out_vis["timesteps"],
                global_step=global_step,
                key=f"eval_image{suff}/t_steps_nfe{nfe}",
                experiment=experiment,
            )
            log_t_steps(
                t_steps=out_vis["timesteps"],
                global_step=global_step,
                key=f"t_stats{suff}/nfe{nfe}",
                experiment=experiment,
            )
            log_end_img(
                out_vis["x0_s"],
                out_vis["x0_t"],
                global_step=global_step,
                key=f"vis_stat{suff}/backward_end_inter_nfe{nfe}",
                experiment=experiment,
            )
            for k, v in out_vis.items():
                if k not in NOT_LOG_KEYS:
                    d_res[f"vis_stat{suff}/nfe{nfe}/{k}"] = v.mean().item()

        # ------------------------------------------------------------------ #
        # Test dataset — one full pass per NFE                                #
        # ------------------------------------------------------------------ #
        for nfe in nfe_list:
            log_d: dict = defaultdict(float)
            num_elements = 0
            out_test = None

            for batch in data.test_loader:
                batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
                out_test = gs_wrapper.forward(
                    batch=batch, return_timesteps=False, is_train=False, n_steps_override=nfe
                )
                bs = batch[0].shape[0]
                num_elements += bs
                for k, v in out_test.items():
                    if k not in NOT_LOG_KEYS:
                        log_d[k] += v.mean().item() * bs

            for k, v in log_d.items():
                d_res[f"val_stat{suff}/nfe{nfe}/{k}"] = v / num_elements

            if out_test is not None:
                if "x0_s" not in out_test:
                    out_test["x0_s"] = gs_wrapper.model.decode(out_test["latents_s"])
                log_end_img(
                    out_test["x0_s"],
                    out_test["x0_t"],
                    global_step=global_step,
                    key=f"val_stat{suff}/backward_end_inter_nfe{nfe}",
                    experiment=experiment,
                )

    else:
        # ------------------------------------------------------------------ #
        # Single-NFE (legacy)                                                 #
        # ------------------------------------------------------------------ #
        nfe = gs_wrapper.steps
        out_vis = gs_wrapper.forward(batch=vis_batch, return_timesteps=True, is_train=False)
        if "x0_s" not in out_vis:
            out_vis["x0_s"] = gs_wrapper.model.decode(out_vis["latents_s"])

        log_t_steps_plot(
            t_steps=out_vis["timesteps"],
            global_step=global_step,
            key=f"eval_image{suff}/t_steps_nfe{nfe}",
            experiment=experiment,
        )
        log_t_steps(
            t_steps=out_vis["timesteps"],
            global_step=global_step,
            key=f"t_stats{suff}/nfe{nfe}",
            experiment=experiment,
        )
        log_end_img(
            out_vis["x0_s"],
            out_vis["x0_t"],
            global_step=global_step,
            key=f"vis_stat{suff}/backward_end_inter_nfe{nfe}",
            experiment=experiment,
        )
        for k, v in out_vis.items():
            if k not in NOT_LOG_KEYS:
                d_res[f"vis_stat{suff}/nfe{nfe}/{k}"] = v.mean().item()

        log_d: dict = defaultdict(float)
        num_elements = 0
        out_test = None

        for batch in data.test_loader:
            batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
            out_test = gs_wrapper.forward(batch=batch, return_timesteps=False, is_train=False)
            bs = batch[0].shape[0]
            num_elements += bs
            for k, v in out_test.items():
                if k not in NOT_LOG_KEYS:
                    log_d[k] += v.mean().item() * bs

        for k, v in log_d.items():
            d_res[f"val_stat{suff}/nfe{nfe}/{k}"] = v / num_elements

        if out_test is not None:
            if "x0_s" not in out_test:
                out_test["x0_s"] = gs_wrapper.model.decode(out_test["latents_s"])
            log_end_img(
                out_test["x0_s"],
                out_test["x0_t"],
                global_step=global_step,
                key=f"val_stat{suff}/backward_end_inter_nfe{nfe}",
                experiment=experiment,
            )

    experiment.log_metrics(d_res, step=global_step)

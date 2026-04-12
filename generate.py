import os
import re
from functools import partial

import click
import numpy as np
import PIL.Image
import torch
import tqdm
import yaml
from ml_collections import ConfigDict

from src.gas.models import get_gs_wrapper, load_base_model
from src.gas.sampling_algs import SAMPLING_ALGS
from torch_utils import distributed as dist


def custom_to_np(x: torch.Tensor) -> np.array:
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.numpy()
    return sample


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


@click.command()
@click.option(
    "--config", "config_path", help="", metavar="PATH", type=str, required=True
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--seeds",
    help="Random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="0-63",
    show_default=True,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of sampling steps. When --checkpoint_path is set and create_dataset=False, "
         "accepts a comma-separated list (e.g. '4,5,6') to generate images for multiple NFEs; "
         "requires t_parametrization=ar_model in the config. Each NFE is saved to its own subdir.",
    metavar="INT_OR_LIST",
    type=str,
    required=False,
    default=None,
)
@click.option("--checkpoint_path", help="GS checkpoint path", metavar="PATH", type=str)
@click.option("--create_dataset", help="", metavar="BOOL", type=bool, default=False)
def main(
    config_path,
    outdir,
    seeds,
    max_batch_size,
    num_steps,
    checkpoint_path,
    create_dataset,
    device=torch.device("cuda"),
):
    dist.init()

    # Parse steps: single int or comma-separated list
    if num_steps is not None:
        steps_list = [int(s.strip()) for s in num_steps.split(",")]
    else:
        steps_list = None
    is_multi_nfe = steps_list is not None and len(steps_list) > 1

    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    with open(config_path) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    gs_solver = checkpoint_path is not None
    model_config = config.model
    solver_config = (
        config.student_solver_config if gs_solver else config.teacher_solver_config
    )

    if is_multi_nfe:
        assert gs_solver, "Multi-NFE generation requires --checkpoint_path"
        assert not create_dataset, (
            "Multi-NFE mode only supports image generation (create_dataset must be False)"
        )
        assert solver_config.t_parametrization == "ar_model", (
            "Multi-NFE generation requires t_parametrization=ar_model in student_solver_config"
        )
        max_nfe = max(steps_list)
        solver_config.loss_config.loss_type = "GS"
        solver_config.steps = max_nfe
        solver_config.order = max_nfe
    else:
        single_steps = steps_list[0] if steps_list else None
        assert (single_steps is None) != (
            solver_config.steps is None
        ), "Steps should be specified in one and only one of both generate script and solver config"

    # Load base model.
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale
    model = load_base_model(model_config, device)

    if is_multi_nfe:
        gs_wrapper = get_gs_wrapper(model, solver_config)
        gs_wrapper.load_checkpoint(checkpoint_path=checkpoint_path)
    elif gs_solver:
        solver_config.loss_config.loss_type = "GS"
        solver_config.steps = single_steps
        solver_config.order = single_steps
        gs_wrapper = get_gs_wrapper(model, solver_config)
        gs_wrapper.load_checkpoint(checkpoint_path=checkpoint_path)
        sampler_fn = partial(gs_wrapper.student_sampler_fn, decode=True)
    else:
        if single_steps is not None:
            solver_config.steps = single_steps
        sampler_fn = SAMPLING_ALGS[model_config.type]
        sampler_fn = partial(sampler_fn, model=model, solver_config=solver_config)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    shape = [None, model.image_channels, model.image_size, model.image_size]

    # ------------------------------------------------------------------ #
    # Multi-NFE: generate full seed set for each NFE into subdirs         #
    # ------------------------------------------------------------------ #
    if is_multi_nfe:
        images_base = os.path.join(outdir, "images")
        for nfe in steps_list:
            nfe_outdir = os.path.join(images_base, str(nfe))
            os.makedirs(nfe_outdir, exist_ok=True)
            nfe_sampler_fn = partial(gs_wrapper.student_sampler_fn, decode=True, n_steps=nfe)

            dist.print0(f'Generating {len(seeds)} images (nfe={nfe}) to "{nfe_outdir}"...')
            for batch_seeds in tqdm.tqdm(
                rank_batches, unit="batch", disable=(dist.get_rank() != 0)
            ):
                torch.distributed.barrier()

                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                shape[0] = batch_size
                rnd = StackedRandomGenerator(device, batch_seeds)
                noise = rnd.randn(shape, device=device)

                condition = None
                if model_config.conditional:
                    condition = model.iterate_condition(batch_seeds.tolist())

                with torch.no_grad():
                    _, images = nfe_sampler_fn(noise=noise, condition=condition)

                if model_config.type == "EDM":
                    images_np = (
                        (images * 127.5 + 128)
                        .clip(0, 255)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .cpu()
                        .numpy()
                    )
                else:
                    images_np = custom_to_np(images)

                for seed, image_np in zip(batch_seeds, images_np):
                    PIL.Image.fromarray(image_np, "RGB").save(
                        os.path.join(nfe_outdir, f"{seed:06d}.png")
                    )

            torch.distributed.barrier()
            dist.print0(f"Done (nfe={nfe}, n={len(seeds)}).")

        dist.print0("All done.")
        return

    # ------------------------------------------------------------------ #
    # Single-NFE (original behaviour)                                      #
    # ------------------------------------------------------------------ #
    if create_dataset:
        synt_dir = os.path.join(outdir, "dataset")
        os.makedirs(synt_dir, exist_ok=True)

    outdir = os.path.join(outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(
        rank_batches, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()

        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        shape[0] = batch_size

        rnd = StackedRandomGenerator(device, batch_seeds)
        noise = rnd.randn(shape, device=device)

        condition = None
        if model_config.conditional:
            condition = model.iterate_condition(batch_seeds.tolist())

        with torch.no_grad():
            latents, images = sampler_fn(noise=noise, condition=condition)

        if create_dataset:
            latents = [None] * batch_size if latents is None else latents
            condition = [None] * batch_size if condition is None else condition

            dataset = {
                "noise": noise.detach().cpu(),
                "latents": (
                    latents.detach().cpu()
                    if isinstance(latents, torch.Tensor)
                    else latents
                ),
                "images": images.detach().cpu(),
                "condition": (
                    condition.detach().cpu()
                    if isinstance(condition, torch.Tensor)
                    else condition
                ),
            }
            torch.save(dataset, os.path.join(synt_dir, f"{batch_seeds[0]}.pt"))

        if model_config.type == "EDM":
            images_np = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
        else:
            images_np = custom_to_np(images)

        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(outdir, f"{seed:06d}.png")
            PIL.Image.fromarray(image_np, "RGB").save(image_path)

    torch.distributed.barrier()
    dist.print0("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

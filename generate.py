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


def _generate_batches(
    seeds_subset,
    sampler_fn,
    model,
    model_config,
    images_dir,
    synt_dir,
    nfe,
    device,
    max_batch_size,
):
    """Distributed generation loop for a list of seeds.

    Args:
        seeds_subset: List of integer seeds to generate.
        sampler_fn: Callable ``(noise, condition) -> (latents, images)``.
        model: Loaded base model (provides image_channels, image_size).
        model_config: Model config dict.
        images_dir: Directory to save PNG images.
        synt_dir: Directory to save .pt dataset files, or None to skip.
        nfe: Step count label stored in the 'n_steps' field of each .pt file.
        device: Torch device.
        max_batch_size: Maximum batch size per generation step.
    """
    os.makedirs(images_dir, exist_ok=True)
    if synt_dir is not None:
        os.makedirs(synt_dir, exist_ok=True)

    if len(seeds_subset) == 0:
        return

    seeds_tensor = torch.as_tensor(seeds_subset)
    num_batches = (
        (len(seeds_subset) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = seeds_tensor.tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    shape = [None, model.image_channels, model.image_size, model.image_size]

    dist.print0(f'Generating {len(seeds_subset)} images (steps={nfe}) to "{images_dir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0)):
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

        if synt_dir is not None:
            latents_save = [None] * batch_size if latents is None else latents
            condition_save = [None] * batch_size if condition is None else condition
            dataset_entry = {
                "noise": noise.detach().cpu(),
                "latents": (
                    latents_save.detach().cpu()
                    if isinstance(latents_save, torch.Tensor)
                    else latents_save
                ),
                "images": images.detach().cpu(),
                "condition": (
                    condition_save.detach().cpu()
                    if isinstance(condition_save, torch.Tensor)
                    else condition_save
                ),
                "n_steps": [nfe] * batch_size,
            }
            torch.save(dataset_entry, os.path.join(synt_dir, f"{batch_seeds[0]}.pt"))

        # Save images
        if model_config.type == "EDM":
            # Saves the batch in EDM style as in https://github.com/NVlabs/edm/blob/main/generate.py
            images_np = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
        else:
            # Saves the batch in LDM style as in https://github.com/CompVis/latent-diffusion/blob/main/scripts/sample_diffusion.py
            images_np = custom_to_np(images)

        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(images_dir, f"{seed:06d}.png")
            PIL.Image.fromarray(image_np, "RGB").save(image_path)

    torch.distributed.barrier()
    dist.print0(f"Done (steps={nfe}, n={len(seeds_subset)}).")


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
    help="Random seeds defining the total sample pool (e.g. 0-2399). "
         "In multi-step mode this range is partitioned across step counts "
         "according to --steps_ratios, with the last --test_size seeds "
         "reserved for the test split.",
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
    help="Step count(s): single int or comma-separated list for multi-step dataset "
         "creation (e.g. '4,5,6,7'). In multi-step mode the seed range is split "
         "across step counts according to --steps_ratios.",
    metavar="INT_OR_LIST",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--steps_ratios",
    help="Comma-separated ratios controlling how the train seeds are distributed "
         "across step counts (e.g. '1,2' for a 1:2 ratio). Order matches --steps. "
         "Defaults to equal ratios.",
    metavar="LIST",
    type=str,
    default=None,
)
@click.option(
    "--test_size",
    help="Number of seeds taken from the END of the seed range and used as test "
         "data for EVERY step count (e.g. test_size=1000 with nfe={4,5} produces "
         "1000 test samples for each). Saved to out/dataset/test/<N>/. Must be "
         "less than the total seed count. Only used with create_dataset=True.",
    metavar="INT",
    type=int,
    default=0,
)
@click.option("--checkpoint_path", help="GS checkpoint path", metavar="PATH", type=str)
@click.option("--create_dataset", help="", metavar="BOOL", type=bool, default=False)
def main(
    config_path,
    outdir,
    seeds,
    max_batch_size,
    num_steps,
    steps_ratios,
    test_size,
    checkpoint_path,
    create_dataset,
    device=torch.device("cuda"),
):
    dist.init()

    # ------------------------------------------------------------------ #
    # Parse steps and ratios                                               #
    # ------------------------------------------------------------------ #
    if num_steps is not None:
        steps_list = [int(s.strip()) for s in num_steps.split(",")]
    else:
        steps_list = [None]

    is_multi_step = len(steps_list) > 1 and steps_list[0] is not None

    if is_multi_step:
        if steps_ratios is not None:
            raw_ratios = [float(r.strip()) for r in steps_ratios.split(",")]
            assert len(raw_ratios) == len(steps_list), (
                f"--steps_ratios must have the same number of entries as --steps "
                f"(got {len(raw_ratios)} vs {len(steps_list)})"
            )
        else:
            raw_ratios = [1.0] * len(steps_list)
        total_r = sum(raw_ratios)
        norm_ratios = [r / total_r for r in raw_ratios]
    else:
        norm_ratios = [1.0]

    # ------------------------------------------------------------------ #
    # Split seeds into train / test portions                              #
    # ------------------------------------------------------------------ #
    all_seeds = list(seeds)

    if create_dataset and is_multi_step and test_size > 0:
        assert test_size < len(all_seeds), (
            f"--test_size ({test_size}) must be less than the total number of seeds "
            f"({len(all_seeds)})"
        )
        test_seeds = all_seeds[-test_size:]
        train_seeds = all_seeds[:-test_size]
    else:
        test_seeds = []
        train_seeds = all_seeds

    # Distribute train seeds by ratio
    train_seeds_per_nfe: dict = {}
    offset = 0
    for i, (nfe, ratio) in enumerate(zip(steps_list, norm_ratios)):
        if i == len(steps_list) - 1:
            n = len(train_seeds) - offset          # remainder goes to last group
        else:
            n = round(ratio * len(train_seeds))
        train_seeds_per_nfe[nfe] = train_seeds[offset: offset + n]
        offset += n

    # Same test seeds for every NFE
    test_seeds_per_nfe: dict = {}
    if test_seeds:
        for nfe in steps_list:
            test_seeds_per_nfe[nfe] = test_seeds

    dist.print0(
        f"Seed distribution — train: "
        + ", ".join(f"nfe={n}: {len(s)}" for n, s in train_seeds_per_nfe.items())
        + (
            " | test: "
            + ", ".join(f"nfe={n}: {len(s)}" for n, s in test_seeds_per_nfe.items())
            if test_seeds_per_nfe else ""
        )
    )

    # ------------------------------------------------------------------ #
    # Rank 0 loads model first                                            #
    # ------------------------------------------------------------------ #
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    with open(config_path) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    gs_solver = checkpoint_path is not None
    model_config = config.model
    solver_config = (
        config.student_solver_config if gs_solver else config.teacher_solver_config
    )

    # For single-step mode: validate that steps is specified exactly once
    if not is_multi_step:
        single_steps = steps_list[0]
        assert (single_steps is None) != (solver_config.steps is None), (
            "Steps should be specified in one and only one of the generate script "
            "and solver config"
        )

    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale
    model = load_base_model(model_config, device)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # ------------------------------------------------------------------ #
    # Generate per step count                                             #
    # ------------------------------------------------------------------ #
    for nfe in steps_list:
        # Set up sampler for this NFE
        if gs_solver:
            solver_config.loss_config.loss_type = "GS"
            if nfe is not None:
                solver_config.steps = nfe
                solver_config.order = nfe
            gs_wrapper = get_gs_wrapper(model, solver_config)
            gs_wrapper.load_checkpoint(checkpoint_path=checkpoint_path)
            sampler_fn = partial(gs_wrapper.student_sampler_fn, decode=True)
        else:
            if nfe is not None:
                solver_config.steps = nfe
            sampler_fn = SAMPLING_ALGS[model_config.type]
            sampler_fn = partial(sampler_fn, model=model, solver_config=solver_config)

        # Output directories
        if is_multi_step:
            train_images_dir = os.path.join(outdir, "images", str(nfe))
            train_synt = os.path.join(outdir, "dataset", str(nfe)) if create_dataset else None
            test_images_dir = os.path.join(outdir, "images", "test", str(nfe))
            test_synt = (
                os.path.join(outdir, "dataset", "test", str(nfe))
                if create_dataset and test_seeds_per_nfe
                else None
            )
        else:
            train_images_dir = os.path.join(outdir, "images")
            train_synt = os.path.join(outdir, "dataset") if create_dataset else None
            test_images_dir = None
            test_synt = None

        # Generate train split
        _generate_batches(
            seeds_subset=train_seeds_per_nfe[nfe],
            sampler_fn=sampler_fn,
            model=model,
            model_config=model_config,
            images_dir=train_images_dir,
            synt_dir=train_synt,
            nfe=nfe,
            device=device,
            max_batch_size=max_batch_size,
        )

        # Generate test split (multi-step only)
        if test_seeds_per_nfe and nfe in test_seeds_per_nfe:
            _generate_batches(
                seeds_subset=test_seeds_per_nfe[nfe],
                sampler_fn=sampler_fn,
                model=model,
                model_config=model_config,
                images_dir=test_images_dir,
                synt_dir=test_synt,
                nfe=nfe,
                device=device,
                max_batch_size=max_batch_size,
            )

    dist.print0("All done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

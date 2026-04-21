import click
import torch
import yaml
from ml_collections import ConfigDict
from torch_ema import ExponentialMovingAverage

from src.gas.models import get_gs_wrapper, load_base_model
from src.gas.synt_data import SyntDataLoaders
from src.gas.utils.random import set_global_seed
from training import train


@click.command()
@click.option(
    "--config",
    metavar="PATH",
    type=str,
    required=True,
    help="Path to config including model, training and dataset info.",
)
@click.option(
    "--loss_type",
    metavar="GS|GAS",
    type=click.Choice(["GS", "GAS"]),
    help="Loss type to train GS.",
)
@click.option(
    "--student_step",
    metavar="INT",
    type=click.IntRange(4, 20),
    default=None,
    help="Number of student steps. Required for single-NFE training; "
         "optional for mixed-NFE AR training (steps come from the dataset).",
)
@click.option(
    "--teacher_pkl",
    metavar="PATH",
    type=str,
    default=None,
    help="Path to teacher pkl dataset.",
)
@click.option(
    "--train_size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=1400,
    help="Size of the training dataset (default: 1400).",
)
def main(
    config: str,
    loss_type: str,
    student_step: int,
    teacher_pkl: str,
    train_size: int,
    device=torch.device("cuda"),
):
    with open(config) as stream:
        config = ConfigDict(yaml.safe_load(stream))

    # Setup dataset config
    dataset_config = config.dataset
    assert (dataset_config.teacher_pkl is None) != (
        teacher_pkl is None
    ), "You should set one and only one of teacher pickles"

    if dataset_config.teacher_pkl is None:
        dataset_config.teacher_pkl = teacher_pkl
    dataset_config.train_size = train_size

    # Setup student solver config
    solver_config = config.student_solver_config
    solver_config.loss_config.loss_type = loss_type

    # Determine steps: CLI arg takes priority; for mixed-NFE AR mode the
    # steps can be left as null in the config (AR model generates varying NFEs at runtime).
    use_mixed_nfe = (
        getattr(solver_config, 'steps_ratios', None) is not None
        and solver_config.t_parametrization == "ar_model"
    )
    # When learn_correctors=True with ar_model, order must be explicitly set in config
    # to the solver order (e.g. 3); do not derive it from NFE count.
    ar_learn_correctors = (
        getattr(solver_config, 'learn_correctors', False)
        and solver_config.t_parametrization == "ar_model"
    )
    if student_step is not None:
        solver_config.steps = student_step
        if not ar_learn_correctors:
            solver_config.order = student_step
    elif use_mixed_nfe:
        # In mixed-NFE mode the AR model handles varying step counts at runtime.
        # solver_config.steps must still be set so GSWrapper can size internal
        # fixed-size parameters (t_couple, c_diff, a_diff) to at least the
        # maximum step count that will be used during training.
        if solver_config.steps is None:
            max_nfe = max(int(k) for k in solver_config.steps_ratios.keys())
            solver_config.steps = max_nfe
        if not ar_learn_correctors:
            # derive order from steps (standard behaviour)
            solver_config.order = solver_config.steps
        # else: order must be set explicitly in config (asserted in GSWrapper)
    else:
        raise click.UsageError(
            "--student_step is required when not using mixed-NFE AR training "
            "(i.e. when student_solver_config.steps_ratios is not set or t_parametrization != 'ar_model')."
        )

    solver_config.student_name = "_".join(
        f"{k}={v}" for k, v in solver_config.items() if k != "loss_config"
    )
    if config.logging.run_name is None:
        config.logging.run_name = (
            f"{solver_config.student_name}_{dataset_config.teacher_pkl}_{loss_type}"
        )

    # Setup dataset
    set_global_seed(config.seed)
    data = SyntDataLoaders(dataset_config, seed=config.seed)

    # Setup model
    model_config = config.model
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale

    base_model = load_base_model(model_config, device)
    gs_wrapper = get_gs_wrapper(base_model, solver_config)

    # Setup training
    optim = torch.optim.Adam(gs_wrapper.parameters(), lr=config.training.lr)
    ema = ExponentialMovingAverage(
        gs_wrapper.parameters(), decay=config.training.ema_decay
    )
    n_iters = config.training.n_iters
    config.training.epoch_num = n_iters // len(data.train_loader) + int(
        n_iters % len(data.train_loader) != 0
    )

    train(
        config=config,
        gs_wrapper=gs_wrapper,
        ema=ema,
        data=data,
        optim=optim,
        device=device,
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

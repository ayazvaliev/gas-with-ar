import lpips
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Optional, Any, List
from torch.nn.functional import interpolate
from torch_ema import ExponentialMovingAverage
from ml_collections import ConfigDict
from tqdm import tqdm

from src.gas.base_model import BaseModel
from src.gas.generalized_solver import GeneralizedSolver
from src.gas.adversarial_module.dist_adv_loss import DistAdversarialTraining
from src.gas.synt_data import SyntDataType
from src.gas.ar.model import ARModel

class GSWrapper(nn.Module):
    """Generalised Solver wrapper. 
    
    This class integrates all the logic needed to train or evaluate 
    the given generative model using Generalised Solver. 
    
    Method `student_sampler_fn` is used to call the sampler.
    Method `forward` is called in the training loop to calculate the losses.
        The model can be trained in both default or adversarial modes.
    
    Attributes:
        model (BaseModel): Underlying model instance wrapped in BaseModel interface.
        solver_config (ConfigDict): Solver configuration dictionary.
        solver (GeneralizedSolver): Generalised Solver instance that is trained/evaluated.
        
        loss_fn_vgg (nn.Module): VGG model instance to calculate LPIPS loss.
        adv_loss (DistAdversarialTraining): Adversarial training class instance. 
    """
    
    def __init__(self, model: BaseModel, solver_config: ConfigDict, run_warmup: bool = True):
        """Initialize the Generalised Solver wrapper.

        Args:
            model (BaseModel): Instance of a BaseModel class.
                Its `decode`, `set_condition` methods and
                `model_fn`, `ns` and `t_eps` attributes are used.
            solver_config (ConfigDict): Solver configuration dictionary.
                Must include steps, order, loss_config,
                t_parametrization and use_theory_coef.
            run_warmup (bool): Whether to run AR warmup on init. Set to False
                when a checkpoint will be loaded immediately after construction.
        """
        super().__init__()
        self.model = model
        self.solver_config = solver_config
        self.t_eps = self.model.t_eps
        
        self._loss_fn_vgg = None
        self._ar_out_cache: dict = {}

        # construct loss
        self.loss_config = self.solver_config.loss_config
        assert self.loss_config.loss_type in ["GS", "GAS"]
        if self.loss_config.loss_type == "GAS":
            self.adv_loss = DistAdversarialTraining(self.loss_config)

        # setup solver
        solver = self.get_base_solver()
        self.steps = self.solver_config.steps
        self.order = self.solver_config.order
        self.learn_correctors = getattr(self.solver_config, 'learn_correctors', False)

        # init t steps
        self.eps_mu_offset = 1e-5
        if self.solver_config.t_parametrization == "mu_logit":
            self.mu_logit = nn.Parameter(torch.ones(self.steps - 1), requires_grad=True)
            t_unif = torch.linspace(1., self.t_eps, self.steps + 1).flip(0)
            self.mu_logit.data = self.get_inv_t_steps(t_unif)
            self.act = torch.nn.functional.sigmoid
            use_ar = False
        elif self.solver_config.t_parametrization == "ar_model":
            if self.learn_correctors:
                assert self.order is not None, (
                    "solver_config.order must be explicitly set to the solver order (e.g. 3) "
                    "when learn_correctors=True with ar_model t_parametrization"
                )
            ar_config = solver_config.ar_config
            ar_config.learn_correctors = self.learn_correctors
            ar_config.corrector_order = self.order if self.learn_correctors else 1
            self.ar_model = ARModel(ar_config)
            self.act = lambda x: 0.5 * (torch.nn.functional.softsign(x) + 1)
            use_ar = True
            ar_warmup_cfg = getattr(self.solver_config, 'ar_warmup', None)
            if run_warmup and ar_warmup_cfg is not None and getattr(ar_warmup_cfg, 'enabled', False):
                self._run_ar_warmup()
        else:
            raise NotImplementedError()
        solver.get_time_steps = lambda **kwargs: self.get_t_steps(use_ar=use_ar, **kwargs)

        # init t_couple
        self.t_couple = nn.Parameter(torch.zeros(self.steps), requires_grad=False)
        solver.t_couple = self.t_couple

        # init coef
        # mu_logit: requires_grad follows learn_correctors flag
        # ar_model: always False here; solver attrs are overridden dynamically in get_t_steps
        coef_requires_grad = (
            self.learn_correctors
            and self.solver_config.t_parametrization == "mu_logit"
        )
        for i in range(1, self.order + 1):
            cname, aname = f'c{i}_diff', f'a{i}_diff'

            self.register_parameter(
                param=nn.Parameter(torch.zeros(self.steps), requires_grad=coef_requires_grad),
                name=cname
            )
            self.register_parameter(
                param=nn.Parameter(torch.zeros(self.steps), requires_grad=coef_requires_grad),
                name=aname
            )

            solver.__setattr__(cname, self.__getattr__(cname))
            solver.__setattr__(aname, self.__getattr__(aname))

        # theory coef
        solver.use_theory_coef = self.solver_config.use_theory_coef
        if not solver.use_theory_coef:
            solver.init_coefs(
                steps=self.steps,
                order=self.order,
                timesteps=self.get_t_steps()
            )
        # end init solver
        self.solver = solver

    def train(self, mode: bool = True):
        if mode:
            self._ar_out_cache.clear()
        return super().train(mode)

    # timesteps logic
    def get_t_steps(self, **kwargs) -> torch.Tensor:
        """Get generation timesteps.

        When ``n_steps`` is supplied via kwargs (set by student_sampler_fn for
        mixed-NFE AR training), the AR model generates logits for that step
        count instead of the default ``self.steps``.

        When ``learn_correctors`` is True and ``t_parametrization`` is ``ar_model``,
        the AR model outputs ``1 + 2*order`` values per step. Corrector values are
        extracted and set on the solver as side-effects before sampling begins.
        AR outputs cover steps 0..n_steps-2; step n_steps-1 gets zero correctors.

        In eval mode (inference), AR model outputs are cached by n_steps to avoid
        redundant forward passes across batches.
        """
        if kwargs.get("use_ar", False):
            n_steps = kwargs.get("n_steps", self.steps)

            # Cache AR model output in eval mode: model is frozen, same n_steps
            # always produces the same result, so skip the forward on cache hit.
            use_cache = not self.training
            if use_cache and n_steps in self._ar_out_cache:
                ar_out = self._ar_out_cache[n_steps]
            else:
                # When learning correctors, generate one extra AR step so the
                # last solver step also gets a learned corrector instead of zero.
                ar_num_steps = int(n_steps) if self.learn_correctors else int(n_steps) - 1
                ar_out = self.ar_model(ar_num_steps)
                if use_cache:
                    self._ar_out_cache[n_steps] = ar_out

            if self.learn_correctors:
                # ar_out: [n_steps, 1 + 2*order]
                # first n_steps-1 rows supply timestep logits; all n_steps rows supply correctors
                logits = ar_out[:-1, 0]
                # self.solver may not exist yet during __init__ warmup; skip then
                if hasattr(self, 'solver'):
                    for i in range(1, self.order + 1):
                        setattr(self.solver, f'a{i}_diff', ar_out[:, i])
                        setattr(self.solver, f'c{i}_diff', ar_out[:, self.order + i])
            else:
                logits = ar_out
        else:
            logits = self.mu_logit
        t = self.get_mu_t_steps(logits)
        return t.flip(0)
    
    def get_mu_t_steps(self, mu_logit: torch.Tensor) -> torch.Tensor:
        """Use stick-breaking transform for getting timesteps from logits.
        Timesteps are calculated following Eq. 14 from the GAS paper.
        """
        t_offset = self.t_eps

        mu = self.act(mu_logit)
        mu = mu * (1 - 2 * self.eps_mu_offset) + self.eps_mu_offset

        t_steps = 1 - torch.cumprod(mu, 0)
        t_steps = t_steps * (1 - t_offset) + t_offset
        t_steps = torch.cat(
            [
                torch.zeros_like(t_steps[:1]) + t_offset,
                t_steps,
                torch.ones_like(t_steps[:1])
            ]
        )
        return t_steps
    
    def get_inv_t_steps(self, t_steps) -> torch.Tensor:
        """Function to inverse initialized timesteps."""
        t_steps = t_steps[1:-1]
        t_steps = 1 - (t_steps - self.t_eps) / (1 - self.t_eps)
        t_steps = t_steps / torch.concat([torch.ones_like(t_steps[:1]), t_steps[:-1]])
        t_steps = (t_steps - self.eps_mu_offset) / (1 - 2 * self.eps_mu_offset)

        return t_steps.logit()
    
    '''
    @torch.no_grad()
    def get_timesteps_for_n(self, n_steps: int) -> torch.Tensor:
        """Return solver timesteps for a specific n_steps value.

        Only meaningful in AR mode (t_parametrization='ar_model').
        Falls back to the default self.steps in mu_logit mode.
        """
        return self.solver.get_time_steps(n_steps=n_steps)
    '''

    # utilities
    def _run_ar_warmup(self) -> None:
        """Pre-train AR model weights so initial timesteps approximate uniform spacing.

        For each NFE, the target is the interior points of a uniform grid over
        [t_eps, 1.0]. In mixed-NFE mode, gradients from all NFEs are accumulated
        before each optimizer step so the model learns all step counts jointly.
        """
        warmup_cfg = self.solver_config.ar_warmup
        n_iters = warmup_cfg.n_iters
        lr = warmup_cfg.lr

        steps_ratios = getattr(self.solver_config, 'steps_ratios', None)
        nfe_list = sorted(int(k) for k in steps_ratios.keys()) if steps_ratios is not None else [self.steps]

        warmup_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ar_model.to(warmup_device)
        targets = {
            nfe: torch.linspace(1.0, self.t_eps, nfe + 1)[1:-1].to(warmup_device)
            for nfe in nfe_list
        }

        optim = torch.optim.Adam(self.ar_model.parameters(), lr=lr)

        print(f"\nAR warmup started: {n_iters} iters, lr={lr}, NFEs={nfe_list}, device={warmup_device}")
        pbar = tqdm(range(n_iters), desc="AR warmup", dynamic_ncols=True)
        final_loss = 0.0

        for _ in pbar:
            optim.zero_grad()
            total_loss = 0.0
            for nfe in nfe_list:
                t = self.get_t_steps(use_ar=True, n_steps=nfe)
                loss = F.mse_loss(t[1:-1], targets[nfe])
                loss.backward()
                total_loss += loss.item()
            optim.step()
            final_loss = total_loss
            pbar.set_postfix(loss=f"{total_loss:.6f}")

        print(f"AR warmup done. Final loss: {final_loss:.6f}\n")
        self.ar_model.to("cpu")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads EMA parameters checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        ema = ExponentialMovingAverage(self.parameters(), 0.1)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(self.parameters())

    def parameters(self) -> List[nn.parameter.Parameter]:
        """Returns list of specified solver and wrapper parameters."""
        return list(p for p in super().parameters() if p.requires_grad)

    @property
    def loss_fn_vgg(self):
        if self._loss_fn_vgg is None:
            self._loss_fn_vgg = lpips.LPIPS(net='vgg').requires_grad_(False).eval().to(self.model.device)
        return self._loss_fn_vgg

    def interpolate_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Utility function to resize images for LPIPS calculation."""
        return interpolate(x, size=224, mode='bilinear').clip(-1., 1.)

    # solvers
    def get_base_solver(self) -> GeneralizedSolver:
        """Initialises Generalized Solver from model_fn 
        and noise scheduler of the BaseModel instance.
        """
        solver = GeneralizedSolver(
            model_fn=self.model.model_fn,
            noise_schedule=self.model.ns,
        )
        return solver

    def student_sampler_fn(
        self,
        noise: torch.Tensor,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Calls `sample` method of the Generalised Solver.

        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
            n_steps (int, optional): Override the default step count (for AR mixed-NFE
                training). When provided, the AR model generates logits for this many steps.

        Returns:
            None: A placeholder for consistency with latent models.
            torch.tensor: Sampled images
        """
        steps = n_steps if n_steps is not None else self.steps
        images = self.solver.sample(x=noise, steps=steps, order=steps)
        return None, images

    def _run_student_mixed_nfe(
        self,
        noise: torch.Tensor,
        n_steps_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run student sampler for a batch with per-sample n_steps.

        Samples within each unique n_steps group are processed together, then
        reconstructed in the original batch order. Gradient flow is preserved.
        """
        unique_steps = torch.unique(n_steps_batch)
        batch_order: List[int] = []
        group_imgs: List[torch.Tensor] = []

        for n in unique_steps:
            mask = (n_steps_batch == n).nonzero(as_tuple=True)[0]
            _, student_imgs_n = self.student_sampler_fn(noise[mask], n_steps=n.item())
            group_imgs.append(student_imgs_n)
            batch_order.extend(mask.tolist())

        all_student = torch.cat(group_imgs, dim=0)
        restore_order = torch.argsort(
            torch.tensor(batch_order, device=noise.device)
        )
        return all_student[restore_order]

    def _sample_n_steps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample per-sample NFE assignments from solver_config.steps_ratios."""
        steps_ratios = self.solver_config.steps_ratios
        nfes = sorted(int(k) for k in steps_ratios.keys())
        weights = torch.tensor(
            [float(steps_ratios[str(n)]) for n in nfes], dtype=torch.float32
        )
        indices = torch.multinomial(weights, num_samples=batch_size, replacement=True)
        return torch.tensor(nfes, dtype=torch.long)[indices].to(device)

    # training function
    def forward(
        self,
        batch: SyntDataType,
        return_timesteps: bool = False,
        is_train: bool = True,
        n_steps_override: Optional[int] = None,
    ) -> dict:
        """Forward function used in training loop. Evaluates solver and calculates losses.

        Args:
            batch (SyntDataType): Dataset tuple (noise, images, latents, condition).
            return_timesteps (bool): Flag whether to return timesteps of the current step.
            is_train (bool): Flag whether forward is called in the train loop.
            n_steps_override (int, optional): Fix all samples to this NFE (used in
                evaluation to assess a specific step count on the full test set).

        Returns:
            dict: Dictionary of all losses and model outputs.
                Has `loss_total` key as a weighted sum of adversarial and distillation losses.
        """
        noise, images = batch[0], batch[1]

        steps_ratios = getattr(self.solver_config, 'steps_ratios', None)
        use_mixed_nfe = (
            steps_ratios is not None
            and self.solver_config.t_parametrization == "ar_model"
        )

        d = {}
        if return_timesteps:
            nfe_for_log = n_steps_override if n_steps_override is not None else self.steps
            d['timesteps'] = self.solver.get_time_steps(n_steps=nfe_for_log)

        if use_mixed_nfe:
            if n_steps_override is not None:
                n_steps_batch = torch.full(
                    (noise.shape[0],), n_steps_override,
                    dtype=torch.long, device=noise.device,
                )
            else:
                n_steps_batch = self._sample_n_steps(noise.shape[0], noise.device)
            student_images = self._run_student_mixed_nfe(noise, n_steps_batch)
        else:
            _, student_images = self.student_sampler_fn(noise)

        d['loss_l1'] = torch.abs(student_images - images).mean((1, 2, 3))
        d['loss_l2'] = torch.square(student_images - images).mean((1, 2, 3))

        d['x0_s'] = self.interpolate_lpips(student_images)
        d['x0_t'] = self.interpolate_lpips(images)

        d['loss_lpips'] = self.loss_fn_vgg(d['x0_s'], d['x0_t']).flatten(0)

        if self.loss_config.loss_type == 'GAS':
            # disctiminator step optim
            with torch.no_grad():
                _, student_images_disc = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_images_disc,
                RealSamples=images,
                is_train=is_train
            )
            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step optim
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_images,
                RealSamples=images
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"""
                Shape of generator loss is not equal to distillation loss shape. 
                ({d['gen_loss_adv'].shape} vs {d[self.loss_config.loss_key].shape}).
            """

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d
    
    
class GSWrapperLatent(GSWrapper):
    """Generalised Solver wrapper adapted for latent models."""
    def __init__(self, model: nn.Module, solver_config: ConfigDict, run_warmup: bool = True):
        super().__init__(model=model, solver_config=solver_config, run_warmup=run_warmup)

    def student_sampler_fn(
        self,
        noise: torch.Tensor,
        decode: bool = False,
        condition: Any = None,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calls `sample` method of the Generalised Solver.

        Args:
            noise (torch.Tensor): An initial noise tensor to start sampling process from.
            decode (bool): If True, decode latents to images.
            condition: Optional conditioning.
            n_steps (int, optional): Override step count for AR mixed-NFE training.

        Returns:
            torch.Tensor: Predicted latents that are the direct output of the model.
            Optional[torch.Tensor]: Predicted images (decoded latents).
                Not None if decode flag is set True.
        """
        images = None
        if condition is not None:
            self.model.set_condition(condition)

        steps = n_steps if n_steps is not None else self.steps
        latents = self.solver.sample(x=noise, steps=steps, order=steps)

        if decode:
            images = self.model.decode(latents)

        return latents, images

    def _run_student_mixed_nfe(
        self,
        noise: torch.Tensor,
        n_steps_batch: torch.Tensor,
        condition: Any = None,
    ) -> torch.Tensor:
        """Run student sampler for a latent-model batch with per-sample n_steps."""
        unique_steps = torch.unique(n_steps_batch)
        batch_order: List[int] = []
        group_latents: List[torch.Tensor] = []

        for n in unique_steps:
            mask = (n_steps_batch == n).nonzero(as_tuple=True)[0]
            cond_n = None
            if condition is not None:
                if isinstance(condition, torch.Tensor):
                    cond_n = condition[mask]
                else:
                    cond_n = [condition[i] for i in mask.tolist()]
            latents_n, _ = self.student_sampler_fn(noise[mask], condition=cond_n, n_steps=n.item())
            group_latents.append(latents_n)
            batch_order.extend(mask.tolist())

        all_latents = torch.cat(group_latents, dim=0)
        restore_order = torch.argsort(
            torch.tensor(batch_order, device=noise.device)
        )
        return all_latents[restore_order]

    def forward(
        self,
        batch: SyntDataType,
        return_timesteps: bool = False,
        is_train: bool = True,
        n_steps_override: Optional[int] = None,
    ) -> dict:
        noise, images, latents, condition = batch[0], batch[1], batch[2], batch[3]

        steps_ratios = getattr(self.solver_config, 'steps_ratios', None)
        use_mixed_nfe = (
            steps_ratios is not None
            and self.solver_config.t_parametrization == "ar_model"
        )

        d = {}
        if return_timesteps:
            nfe_for_log = n_steps_override if n_steps_override is not None else self.steps
            d['timesteps'] = self.solver.get_time_steps(n_steps=nfe_for_log)

        if use_mixed_nfe:
            if n_steps_override is not None:
                n_steps_batch = torch.full(
                    (noise.shape[0],), n_steps_override,
                    dtype=torch.long, device=noise.device,
                )
            else:
                n_steps_batch = self._sample_n_steps(noise.shape[0], noise.device)
            student_latents = self._run_student_mixed_nfe(noise, n_steps_batch, condition=condition)
        else:
            student_latents, _ = self.student_sampler_fn(
                noise,
                condition=condition
            )

        d['loss_l1_latents'] = torch.abs(latents - student_latents).mean((1, 2, 3))
        d['loss_l2_latents'] = torch.square(latents - student_latents).mean((1, 2, 3))
        d['x0_t'] = self.interpolate_lpips(images)
        d['latents_s'] = student_latents

        if self.loss_config.loss_type == "GAS":
            with torch.no_grad():
                student_latents_disc, _ = self.student_sampler_fn(
                    torch.randn_like(noise)
                )
            res = self.adv_loss.discriminator_step(
                FakeSamples=student_latents_disc,
                RealSamples=latents,
                is_train=is_train
            )

            d['dis_loss_adv'] = res[0]
            d['dis_scores_fake'] = res[1]
            d['dis_signs_fake'] = res[1].sign()
            d['dis_r1'] = res[2]
            d['dis_r2'] = res[3]

            # generator step
            loss_adv, res = self.adv_loss.AccumulateGeneratorGradients(
                FakeSamples=student_latents,
                RealSamples=latents
            )
            d['gen_loss_adv'] = loss_adv
            d['gen_fake_gen'] = res[1]
            d['gen_signs_fake'] = res[1].sign()

            assert d['gen_loss_adv'].shape == d[self.loss_config.loss_key].shape, f"SHAPE = {d['gen_loss_adv'].shape}, {d[self.loss_config.loss_key].shape}"

        d['loss_total'] = self.loss_config.disc_weight * d.get('gen_loss_adv', 0.) + d[self.loss_config.loss_key]

        return d
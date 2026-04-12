import torch
import pickle
from collections import Counter, defaultdict
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Optional, Union, List, Tuple, Dict

SyntDataType = Tuple[
    torch.Tensor, torch.Tensor,
    Optional[torch.Tensor],
    Optional[Union[torch.Tensor, List[str]]],
    Optional[torch.Tensor],  # n_steps per sample
]


class SyntDataset(Dataset):
    """Dataset class.
    Expects dataset in format as done in generate.py file.
    Optionally contains 'n_steps' field for mixed-NFE training.
    """

    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as fp:
            self.data = pickle.load(fp)

        self.noise_key = 'noise'
        self.images_key = 'images'
        self.latent_key = 'latents'
        self.condition_key = 'condition'
        self.n_steps_key = 'n_steps'

    def __len__(self):
        return len(self.data[self.images_key])

    def __getitem__(self, idx):
        n_steps = None
        if self.n_steps_key in self.data:
            raw = self.data[self.n_steps_key]
            n_steps = int(raw[idx]) if not isinstance(raw[idx], torch.Tensor) else raw[idx]

        return (
            self.data[self.noise_key][idx],
            self.data[self.images_key][idx],
            self.data[self.latent_key][idx],
            self.data[self.condition_key][idx],
            n_steps,
        )

    def get_indices_by_n_steps(self) -> Dict[int, List[int]]:
        """Returns a dict mapping n_steps value -> list of dataset indices."""
        if self.n_steps_key not in self.data:
            return {}
        result = defaultdict(list)
        for idx, n in enumerate(self.data[self.n_steps_key]):
            result[int(n)].append(idx)
        return dict(result)


def _balanced_vis_indices(by_steps: Dict[int, List[int]], size_vis: int) -> List[int]:
    """Pick ``size_vis`` indices evenly spread across n_steps groups.

    Indices are taken from the front of each group (may overlap with train).
    """
    n_groups = len(by_steps)
    vis_per = max(1, size_vis // n_groups)
    result = []
    for n in sorted(by_steps.keys()):
        result.extend(by_steps[n][:vis_per])
    return result


class MixedNFEBatchSampler(Sampler):
    """Batch sampler producing batches that mix samples from multiple n_steps groups.

    Each batch contains samples from every group in proportions defined by
    ``steps_ratios``.  Groups with fewer samples than the largest group are
    **cycled** (reshuffled and reused) so that the total number of batches is
    driven by the *largest* group::

        n_batches = max(len(group) // per_group_count for each group)

    This matches the expected behaviour described in the project docs:
    e.g. if NFE=4 has 16 samples and NFE=5 has 8, with batch_size=8 (4 per
    group), there are 4 batches; batch 2 reuses the first 4 samples of NFE=5.

    Args:
        indices_by_steps: Mapping {n_steps: list of Subset-local indices}.
        steps_ratios: Mapping {n_steps: ratio}.  Normalised internally.
        batch_size: Total batch size.
    """

    def __init__(
        self,
        indices_by_steps: Dict[int, List[int]],
        steps_ratios: Dict[int, float],
        batch_size: int,
    ):
        self.indices_by_steps = indices_by_steps
        self.batch_size = batch_size

        # Normalise ratios
        total = sum(steps_ratios.values())
        self.steps_ratios = {int(k): v / total for k, v in steps_ratios.items()}

        # Per-group batch sizes (at least 1, rounded proportionally, sum == batch_size)
        raw_counts = {n: max(1, round(r * batch_size)) for n, r in self.steps_ratios.items()}
        diff = batch_size - sum(raw_counts.values())
        if diff != 0:
            largest = max(raw_counts, key=raw_counts.get)
            raw_counts[largest] += diff
        self.per_steps_count: Dict[int, int] = raw_counts

        # Number of batches driven by the LARGEST group
        self.n_batches = max(
            len(self.indices_by_steps.get(n, [])) // count
            for n, count in self.per_steps_count.items()
        )

    def __iter__(self):
        # Initial shuffle for each group
        shuffled: Dict[int, List[int]] = {
            n: [idxs[i] for i in torch.randperm(len(idxs)).tolist()]
            for n, idxs in self.indices_by_steps.items()
        }
        pos: Dict[int, int] = {n: 0 for n in self.indices_by_steps}

        for _ in range(self.n_batches):
            batch: List[int] = []
            for n, count in self.per_steps_count.items():
                grp = shuffled[n]
                grp_len = len(grp)
                for _ in range(count):
                    if pos[n] >= grp_len:
                        # Exhausted — reshuffle and cycle
                        shuffled[n] = [grp[i] for i in torch.randperm(grp_len).tolist()]
                        grp = shuffled[n]
                        pos[n] = 0
                    batch.append(grp[pos[n]])
                    pos[n] += 1
            yield batch

    def __len__(self):
        return self.n_batches


class SyntDataLoaders:
    """Synthetic dataset loaders class.

    Class containing all required dataloaders for GS/GAS training:
    train and test loaders, batch for visualisation.

    In mixed-NFE mode (``config.steps_ratios`` is set):
    - Train: first ``train_size`` samples of the dataset.
    - Validation: last ``validation_size`` samples.  collate.py organises the
      dataset so these are the interleaved test samples with equal NFE counts
      — this is verified by an assertion.
    - ``vis_batch`` contains equal numbers of samples per n_steps group.
    - Train loader uses :class:`MixedNFEBatchSampler` (with cycling).

    Attributes:
        train_loader: Dataloader with train data subset.
        test_loader: Dataloader with validation data subset.
        vis_batch: Fixed visualisation batch (balanced per NFE when available).
    """

    def __init__(self, config: ConfigDict):
        self.config = config

        dataset = SyntDataset(dataset_path=self.config.teacher_pkl)
        by_steps = dataset.get_indices_by_n_steps()

        steps_ratios = getattr(config, 'steps_ratios', None)
        use_mixed_nfe = steps_ratios is not None and len(steps_ratios) > 0 and bool(by_steps)

        # ---- Determine train / val sizes -------------------------------- #
        val_size = self.config.validation_size
        train_size = self.config.train_size

        # Cap train_size to available data
        max_train = len(dataset) - val_size
        if train_size is None or train_size > max_train:
            train_size = max_train

        assert train_size > 0, (
            f"No training samples available (dataset={len(dataset)}, val_size={val_size})"
        )

        train_indices = list(range(train_size))
        val_indices = list(range(len(dataset) - val_size, len(dataset)))

        # ---- Validate equal NFE distribution in val (mixed-NFE only) ---- #
        if use_mixed_nfe and dataset.n_steps_key in dataset.data:
            val_n_steps = [int(dataset.data[dataset.n_steps_key][i]) for i in val_indices]
            counts = Counter(val_n_steps)
            assert len(set(counts.values())) == 1, (
                f"Validation set does not have equal NFE distribution: {dict(counts)}. "
                f"Make sure collate.py was run with the test split (dataset/test/<N>/ dirs) "
                f"so that equal-count interleaved test samples are at the end of the pkl."
            )

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, val_indices)

        # ---- Train loader ----------------------------------------------- #
        if use_mixed_nfe:
            # Map original indices -> local Subset indices for the train subset
            orig_to_local = {orig: local for local, orig in enumerate(train_indices)}
            local_by_steps = {
                n: [orig_to_local[orig] for orig in idxs if orig in orig_to_local]
                for n, idxs in by_steps.items()
            }
            batch_sampler = MixedNFEBatchSampler(
                indices_by_steps=local_by_steps,
                steps_ratios=dict(steps_ratios),
                batch_size=self.config.batch_size,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.config.num_workers,
                collate_fn=self.collate_fn,
            )

        # ---- Test loader ------------------------------------------------- #
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.validation_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )

        # ---- vis_batch: balanced per n_steps group ----------------------- #
        if use_mixed_nfe:
            vis_indices = _balanced_vis_indices(by_steps, self.config.size_vis)
        else:
            vis_indices = list(range(min(self.config.size_vis, len(train_dataset))))

        vis_dataset = torch.utils.data.Subset(dataset, vis_indices)
        self.vis_batch = next(
            iter(
                DataLoader(
                    vis_dataset,
                    batch_size=len(vis_indices),
                    shuffle=False,
                    collate_fn=self.collate_fn,
                )
            )
        )

        nfe_info = sorted(by_steps.keys()) if by_steps else "N/A"
        print(f"""
            -------------- Dataloader info --------------
            \tUse latents       = {self.config.use_latents}
            \tUse condition     = {self.config.use_condition}
            \tUse mixed NFE     = {use_mixed_nfe}
            \tNFE groups        = {nfe_info}
            \tTrain size        = {len(train_dataset)}
            \tVal size          = {len(test_dataset)}
            \tVis batch size    = {len(vis_indices)}
            \tlen(train_loader) = {len(self.train_loader)}
            \tlen(test_loader)  = {len(self.test_loader)}
        """)

    def collate_fn(self, batch: Tuple[SyntDataType]) -> SyntDataType:
        """Collates synthetic dataset from teacher pickle into batch.

        Returns:
            5-tuple: (noise, images, latents, condition, n_steps_tensor).
            n_steps_tensor is None when the dataset has no 'n_steps' field.
        """
        noise, images, latents, condition, n_steps = zip(*batch)

        noise = torch.stack(noise)
        images = torch.stack(images)
        latents = torch.stack(latents) if self.config.use_latents else None

        if self.config.use_condition:
            condition = (
                torch.stack(condition)
                if isinstance(condition[0], torch.Tensor)
                else list(condition)
            )
        else:
            condition = None

        if n_steps[0] is not None:
            n_steps_tensor = torch.tensor(list(n_steps), dtype=torch.long)
        else:
            n_steps_tensor = None

        return noise, images, latents, condition, n_steps_tensor

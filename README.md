# Step-agnostic parametrization of solver parameters and time discretization

## Table of Contents
- [Setup Environment](#setup-environment)
- [Download Pretrained Models and FID Reference Sets](#download-pretrained-models-and-fid-reference-sets)
- [Generating Teachers Data](#generating-and-collating-teachers-data)
- [Training Baseline and AR model](#training-baseline-and-ar-model)
- [Evaluating final models](#evaluating-final-models)
- [Datasets](#datasets)
- [Baselines](#baselines)
- [References](#references)
---

## Setup Environment

See [requirements.yml](requirements.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:

```.bash
conda env create -f requirements.yml -n gas
conda activate gas
```


## Download Pretrained Models and FID Reference Sets

All necessary data will be automatically downloaded by the script. Note that this process may take some time. If you wish to skip certain downloads, you can comment out the corresponding lines in the script.

```.bash
bash scripts/downloads.sh
```

## Generating And Collating Teachers Data 

Before training **GS/GAS**, we first need to generate teacher data. To get a batch of images using a teacher solver, run:

```.bash
# Generate 64 images and save them as out/*.png
python generate.py --config=configs/edm/cifar10.yaml \
	--outdir=out \
	--seeds=00000-2399 \
	--batch=64 \
	--create_dataset=True
```

To prepare generated data for training, run:
```.bash
python collate.py --synt_dir=out/dataset --out_pkl=data/teachers/edm/cifar10/dataset.pkl --num_samples 2400
```

## Training Baseline and AR model
To train baseline GS for with frozen correctors for solver parameters, run:
```.bash
python main.py --config=configs/edm/cifar10.yaml \
	--loss_type=GS --student_step=<NFE>
```

To train AR model, run:
```.bash
python main.py --config=configs/edm/cifar10_ar.yaml \
	--loss_type=GS --student_step=<NFE>
```

## Evaluating final models
To compute Fréchet inception distance (FID) for a given solver, generate 50k samples of random images and then compare them against the dataset reference statistics using `fid.py`:

To sample using final model's checkpoint, run:
```.bash
python generate.py \
	--config=<CONFIG_NAME> \
	--outdir=data/outputs/cifar10 \
	--seeds=50000-99999 \
	--batch=1024 \
	--steps=<NFE> \
	--checkpoint_path=<CKPT_PATH>
```

To evaluate FID, run:
```.bash
torchrun --standalone --nproc_per_node=1 fid.py calc \
	--images=data/outputs/cifar10 \
	--ref=fid-refs/edm/cifar10-32x32.npz
```

The command can be parallelized across multiple GPUs by adjusting `--nproc_per_node`. The `fid.py calc` typically takes 1-3 minutes in practice. See python `fid.py --help` for the full list of options.


## Datasets

The teacher data is available at [Hugging Face Hub](https://huggingface.co/datasets/bayes-group-diffusion/GAS-teachers).

| Dataset | Hugging Face Hub 
| :-- | :-- 
| CIFAR-10 | [50k samples link](https://huggingface.co/datasets/bayes-group-diffusion/GAS-teachers/blob/main/edm/cifar10/dataset.pkl)

## Baselines
| Baseline Model | GitHub Link
| :-- | :-- 
| Generalized Solver (GS) | [official implementation](https://github.com/3145tttt/GAS)

## References
[A. Oganov, I. Bykov, E. Neudachina, M. Aliev, A. Tolmachev, A. Sidorov, A. Zuev, A. Okhotin, D. Rakitin, and A. Alanov, “Improving discretization of diffusion ODEs via generalized adversarial solver,” 2025.](https://arxiv.org/abs/2510.17699v1)
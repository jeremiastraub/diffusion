# Representation Learning with (Latent) Diffusion Models
This repository contains the code for the Master thesis "Representation Learning with Diffusion Models".

## Requirements
Checkout ``environment.yaml`` for suitable package versions or directly create and activate a [conda](https://conda.io/) environment via
```bash
conda env create -f environment.yaml
conda activate diffusion
```

🚧 WIP – fix environment

## Pretrained Models
For now, only the checkpoints for those LDMs, LRDMs, t-LRDMs trained on LSUN-Churches are available for download.

You can download all checkpoints via [https://k00.fr/representationDM](https://k00.fr/representationDM). Note that the models trained in a reduced latent space also require the corresponding ``first_stage`` model.

🚧 WIP – corresponding configs

### LDMs

🚧 WIP

### LRDMs

🚧 WIP

### Style-Shape Separating LRDM

🚧 WIP

## Train your own Models

### Data preparation

For downloading and preparing the LSUN-Churches dataset, proceed as described in the [latent-diffusion](https://github.com/CompVis/latent-diffusion#lsun) repository.

### Model Training

Logs and checkpoints for trained models are saved to ``logs/<START_DATE_AND_TIME>_<config-name>``.

Various training configuration files are available in ``configs/``. Models can be trained by running
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/<path-to-config>.yaml -t --gpus 0, -n <name>
```
where ``<name>`` is a custom name of the corresponding log-directory (optional).

## Comments
* The implementation is based on [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).

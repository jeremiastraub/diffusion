""""""
import numpy as np
import torch

import argparse, os, sys, glob, math, time

from omegaconf import OmegaConf
from cutlit import instantiate_from_config, resolve_based_on
from torch.utils.data.dataloader import default_collate

# -----------------------------------------------------------------------------


def bchw_to_st(x):
    x = x.detach().cpu().numpy().transpose(0,2,3,1)
    return 0.5 * (x + 1.)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        help="path to config",
        const=True,
        default="",
    )
    return parser


def get_data(config):
    data = instantiate_from_config(config)
    data.prepare_data()
    data.setup()
    return data


def load_model(config, gpu, eval_mode):
    model = instantiate_from_config(config.model)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return model


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    config = OmegaConf.load(opt.config)
    config = resolve_based_on(config)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)

    gpu = True
    eval_mode = True

    model = load_model(config, gpu, eval_mode)

    print("Preparation done. Start sampling ...")
    time_start = time.time()
    sample = model.sample(shape=(1, 16, 16, 16), sampling_method="ode")
    time_end = time.time()
    print(f"Sampling time: {time_end-time_start:.1f}")

    print(torch.min(sample), torch.max(sample), torch.std(sample))


""""""
import numpy as np
import torch

import argparse, os, sys, glob, math, time

from omegaconf import OmegaConf
import streamlit as st
from cutlit import instantiate_from_config, resolve_based_on
from torch.utils.data.dataloader import default_collate

# -----------------------------------------------------------------------------


def bchw_to_st(x):
    x = x.detach().cpu().numpy().transpose(0,2,3,1)
    return 0.5 * (x + 1.)

@torch.no_grad()
def run_reconstruct(model, dsets):
    """

    """
    if len(dsets.datasets) > 1:
        split = st.sidebar.radio("Split", sorted(dsets.datasets.keys()))
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    batch_size = 1
    start_index = st.sidebar.number_input(
        f"Example Index (Max: {len(dset)-batch_size})",
        value=0,
        min_value=0,
        max_value=len(dset)-batch_size,
    )
    indices = list(range(start_index, start_index+batch_size))

    example = default_collate([dset[i] for i in indices])

    # x = model.get_input(example, "image").to(model.device)

    x = example["image"]
    if len(x.shape) == 3:
        x = x[None, ...]
    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    # example = dset[index]["image"][None, ...]
    # x = torch.from_numpy(example).permute(0, 3, 1, 2).to(
    #     memory_format=torch.contiguous_format, device=model.device
    # )

    # x = model.get_input(example, 0).to(model.device)

    st.write(f"Image: {x.shape}")
    st.image(bchw_to_st(x), clamp=True, output_format="PNG")

    z_option = st.radio("How to obtain latent z?", ["use mode", "sample"])

    if st.button("Run"):
        if z_option == "sample":
            sample_posterior = False
            st.write("Sampling from latent distribution...")
        else:
            sample_posterior = True
            st.write("Using the mode of the latent distribution...")

        x = x.to(model.device)
        xrec, _ = model.ae_model(x, sample_posterior=sample_posterior)
        st.write("Image reconstruction: {}".format(xrec.shape))
        st.image(bchw_to_st(xrec), clamp=True, output_format="PNG")


@torch.no_grad()
def run_sample(model):
    """
    """
    sample_option = st.radio(
        "Where to sample from?", ("Score model", "Gaussian")
    )

    if sample_option == "Score model":

        # TODO Configure sampling interactively

        sample_shape = st.slider(
            "number of samples", min_value=1, max_value=16, step=1, value=1
        )

        sampling_method = st.radio("Sample via ...?", ("pc", "ode"))

        if st.button("Run"):
            sample = model.sample(
                sampling_method=sampling_method,
                shape=(sample_shape, 16, 16, 16),
            )

            st.write("Image: {}".format(sample.shape))
            st.image(bchw_to_st(sample), clamp=True, output_format="PNG")

    elif sample_option == "Gaussian":
        latent_std = st.slider(
            "Latent std", min_value=0., max_value=25., value=5., step=0.1
        )

        if st.button("Run"):
            z = torch.randn((1, 16, 16, 16)) * latent_std
            z = z.to(model.device)
            dec = model.ae_model.decode(z)

            st.write("Image: {}".format(dec.shape))
            st.image(bchw_to_st(dec), clamp=True, output_format="PNG")


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

@st.cache
def get_data(config):
    data = instantiate_from_config(config)
    data.prepare_data()
    data.setup()
    return data

@st.cache(allow_output_mutation=True)
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

    st.sidebar.text("Options")

    # gpu = st.sidebar.checkbox("GPU", value=True)
    gpu = True
    # eval_mode = st.sidebar.checkbox("eval mode", value=True)
    eval_mode = True


    model = load_model(config, gpu, eval_mode)

    run_option = st.radio("Choose the task", ("Sample", "Reconstruct"))

    if run_option == "Reconstruct":
        # get data
        dsets = get_data(config.data)   # calls data.config ...
        # dset = dsets.datasets[list(dsets.datasets.keys())[0]]
        run_reconstruct(model, dsets)

    elif run_option == "Sample":
        run_sample(model)

"""Streamlit script for unconditional image generation

Run via:
    streamlit run unconditional_sampling.py -- --config <path-to-config>
"""
import argparse, os, sys, time, glob

from omegaconf import OmegaConf
import streamlit as st
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from cutlit import instantiate_from_config, resolve_based_on
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------


class stqdm(tqdm):
    """Handling tqdm progress bar in streamlit app.

    From: https://github.com/Wirg/stqdm
    """
    def __init__(
        self,
        iterable=None,
        desc=None,
        total=None,
        leave=True,
        file=None,
        ncols=None,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit="it",
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        lock_args=None,
        nrows=None,
        colour=None,
        gui=False,
        st_container=None,
        backend=False,
        frontend=True,
        **kwargs,
    ):
        if st_container is None:
            st_container = st
        self._backend = backend
        self._frontend = frontend
        self.st_container = st_container
        self._st_progress_bar = None
        self._st_text = None
        super().__init__(
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            gui=gui,
            **kwargs,
        )

    @property
    def st_progress_bar(self) -> st.progress:
        if self._st_progress_bar is None:
            self._st_progress_bar = self.st_container.empty()
        return self._st_progress_bar

    @property
    def st_text(self) -> st.empty:
        if self._st_text is None:
            self._st_text = self.st_container.empty()
        return self._st_text

    def st_display(self, n, total, **kwargs):
        if total is not None and total > 0:
            self.st_text.write(self.format_meter(n, total, **{**kwargs, "ncols": 0}))
            self.st_progress_bar.progress(n / total)

    def display(self, msg=None, pos=None):
        if self._backend:
            super().display(msg, pos)
        if self._frontend:
            self.st_display(**self.format_dict)
        return True

    def st_clear(self):
        if self._st_text is not None:
            self._st_text.empty()
            self._st_text = None
        if self._st_progress_bar is not None:
            self._st_progress_bar.empty()
            self._st_progress_bar = None

    def close(self):
        super().close()
        # Don't remove completed progress bar
        # self.st_clear()


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
    x = model.ae_model.get_input(example, "image")

    st.write(f"Image: {x.shape}")
    st.image(bchw_to_st(x), clamp=True, output_format="PNG")

    z_option = st.radio("How to obtain latent z?", ("sample", "use mode"))

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
def run_score_sampling(model):
    """
    """
    # Sampling configuration
    sampling_method = st.sidebar.radio("Sample via ...?", ("pc", "ode"))

    num_samples = st.sidebar.slider(
        "number of samples", min_value=1, max_value=16, step=1, value=1
    )
    denoise = st.sidebar.checkbox("denoise", value=True)

    # Configure pc-sampler
    if sampling_method == "pc":
        predictor = st.sidebar.radio(
            "Predictor",
            (
                "reverse_diffusion",
                "euler_maruyama",
                "ancestral_sampling",
                "none"
            ),
        )
        corrector = st.sidebar.radio(
            "Corrector", ("langevin", "ald", "none")
        )
        snr = st.sidebar.slider(
            "snr", min_value=0.05, max_value=0.25, value=0.16, step=0.01
        )
        n_steps_each = st.sidebar.slider(
            "n steps each", min_value=1, max_value=5, value=2, step=1
        )
        # continuous = st.radio("continuous", (True, False))
        animate = st.sidebar.checkbox("animate", value=False)
        anim_info = st.empty()
        if animate:
            if num_samples != 1:
                anim_info = st.info(
                    "Animation will be done for a single sample only"
                )
                num_samples = 1

            update_every = st.sidebar.number_input(
                "Update every", min_value=1, max_value=model.sde.N, value=10
            )

    else:
        predictor = None
        corrector = None
        snr = None
        n_steps_each = None
        # continuous = True

    if st.button("Run"):
        # For an accurate timing, would need to time within the actual
        # sampling loops.
        sample_start = time.time()
        if sampling_method == "pc":
            anim_info.empty()
            if animate:
                import imageio
                outvid = "sampling.mp4"
                writer = imageio.get_writer(outvid, fps=25)

            for i, sample in enumerate(
                stqdm(
                    model._sampling_generator(
                        last_only=not animate,
                        sampling_method=sampling_method,
                        shape=(num_samples, 16, 16, 16),
                        predictor=predictor,
                        corrector=corrector,
                        snr=snr,
                        n_steps_each=n_steps_each,
                        noise_removal=denoise,
                    ),
                    desc="Sampling progress",
                    # NOTE +1 due to additional yield statement at the end of
                    #      the sampling generator.
                    total=model.sde.N + 1,
                    frontend=True,
                    backend=True,
                )
            ):
                if animate and i * n_steps_each % update_every == 0:
                    writer.append_data(
                        (bchw_to_st(sample)[0]*255).clip(0, 255).astype(np.uint8)
                    )
                else:
                    # Iterate all the way to the end to get the final sample
                    pass

            if animate:
                writer.close()
                st.video(outvid)

        else:
            sample = model.sample(
                sampling_method=sampling_method,
                shape=(num_samples, 16, 16, 16),
                predictor=predictor,
                corrector=corrector,
                snr=snr,
                n_steps_each=n_steps_each,
                noise_removal=denoise,
            )
        sample_end = time.time()

        st.write(f"Image shape: {sample.shape}")
        st.write(f"Sampling time: < {sample_end-sample_start:.1f} s")
        st.image(bchw_to_st(sample), clamp=True, output_format="PNG")


@torch.no_grad()
def run_gaussian_sampling(model):
    # Sampling from latent Gaussian, just for completeness.
    latent_std = st.sidebar.slider(
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
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser


@st.cache
def get_data(config):
    data = instantiate_from_config(config)
    data.prepare_data()
    data.setup()
    return data


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    return {"model": model}


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(config, ckpt, gpu, eval_mode):
    """

    Args:
        config: The model config
        ckpt: path to checkpoint
        gpu: Whether to use gpu
        eval_mode: Whether to set model to eval mode

    Returns: loaded model, global step
    """
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    if ckpt and "ckpt_path" in config.params:
        st.warning("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None

    model = instantiate_from_config(config)

    if pl_sd["state_dict"] is not None:
        missing, unexpected = model.load_state_dict(
            pl_sd["state_dict"], strict=False
        )
        if missing:
            st.info(f"Missing Keys in State Dict: {missing}")
        if unexpected:
            st.info(f"Unexpected Keys in State Dict: {unexpected}")

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()

    return model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(
            glob.glob(os.path.join(logdir, "configs/*-project.yaml"))
        )
        opt.base = base_configs + opt.base
        opt.resume_from_checkpoint = ckpt

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    configs = [resolve_based_on(cfg) for cfg in configs]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    st.sidebar.text(f"ckpt: {ckpt}")

    gpu = True
    eval_mode = True
    model, global_step = load_model(config.model, ckpt, gpu, eval_mode)

    st.sidebar.text(f"global step: {global_step}")

    run_option = st.radio("Task", ("Sample", "Reconstruct"))
    st.sidebar.text("—————————————————————————————————")
    st.sidebar.text("Options")

    if run_option == "Reconstruct":
        dsets = get_data(config.data) # calls data.config
        run_reconstruct(model, dsets)

    elif run_option == "Sample":
        sample_option = st.sidebar.radio(
            "Where to sample from?", ("Score model", "Gaussian")
        )
        if sample_option == "Score model":
            run_score_sampling(model)
        else:
            run_gaussian_sampling(model)

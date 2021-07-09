"""Streamlit script for unconditional image generation.

Run via:
    streamlit run unconditional_sampling.py -- -r <path-to-logidr>
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
        help=(
            "paths to base configs. Loaded from left-to-right. "
            "Parameters can be overwritten or added with command-line options "
            "of the form `--key value`."
        ),
        default=list(),
    )
    return parser


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
    """Transforms tensor in BCHW format to streamlit-displayable numpy array"""
    x = x.detach().cpu().numpy().transpose(0,2,3,1)
    return 0.5 * (x + 1.)


@st.cache
def get_data(config):
    data = instantiate_from_config(config)
    data.prepare_data()
    data.setup()
    return data


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(config, ckpt, gpu=True, eval_mode=True):
    """"""
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    if ckpt and "ckpt_path" in config.params:
        config.params.ckpt_path = None
        st.warning("Deleted the restore-ckpt path from the config.")

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


@torch.no_grad()
def run_reconstruction(model, dsets):
    """Reconstruct an input image using the autoencoder"""
    st.header("Autoencoder reconstruction")
    # Add extra columns to get the right spacings
    col_left, _, col_mid, col_right, _ = st.beta_columns([3, 1, 3, 3, 2])

    # Input loading configuration
    with col_left:
        if len(dsets) > 1:
            split = st.selectbox("Split", sorted(dsets.keys()))
            dset = dsets[split]
        else:
            dset = next(iter(dsets.values()))

        with st.form(key="input_image_config"):

            index = st.number_input(
                f"Example Index (Max: {len(dset)-1})",
                value=0,
                min_value=0,
                max_value=len(dset)-1,
            )

            button_load_input = st.form_submit_button("Load image")

    # Input loading button action
    if button_load_input:
        batch = default_collate([dset[index]])
        x = model.ae_model.get_input(batch, "image")

        # If input index changed, remove any existing reconstruction image
        if st.session_state.get("input_index", index) != index:
            st.session_state.rec_image = None

        st.session_state.input_index = index
        st.session_state.input_image = x

    # Display current input image + enable image reconstruction
    if st.session_state.get("input_image", None) is not None:
        with col_mid:
            st.subheader("Input")
            st.image(
                bchw_to_st(st.session_state.input_image),
                clamp=True,
                output_format="PNG"
            )

        # Image reconstruction configruation
        with col_left, st.form(key="reconstruction_config"):
            z_option = st.radio(
                "How to obtain latent z?", ("sample", "use mode")
            )
            sample_posterior = z_option == "sample"

            button_reconstruct = st.form_submit_button("Reconstruct")

        # Image reconstruction button action
        if button_reconstruct:
            x = st.session_state.input_image.to(model.device)
            xrec, _ = model.ae_model(x, sample_posterior=sample_posterior)
            st.session_state.rec_image = xrec

        # Display current reconstructed image
        if st.session_state.get("rec_image", None) is not None:
            with col_right:
                st.subheader("Reconstruction")
                st.image(
                    bchw_to_st(st.session_state.rec_image),
                    clamp=True,
                    output_format="PNG"
                )


@torch.no_grad()
def run_sampling(model):
    """Sample from the score-based model"""
    st.header("Sampling")
    col_left, _, col_right, _ = st.beta_columns([5, 1, 4, 2])

    # Sampling configuration
    with col_left:
        sampling_method = st.selectbox(
            "Sampling method",
            (
                "PC (predictor-corrector)",
                "ODE",
                "latent autoencoder prior (without score-model)",
            )
        )

    with col_left, st.form(key="sampling_config"):
        # PC-sampling configuration
        if sampling_method == "PC (predictor-corrector)":

            predictor = st.selectbox(
                "Predictor",
                (
                    "reverse_diffusion",
                    "euler_maruyama",
                    "ancestral_sampling",
                    "none"
                ),
            )
            corrector = st.selectbox(
                "Corrector", ("langevin", "ald", "none")
            )
            n_steps_each = st.slider(
                "corrector steps per predictor step",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
            )
            snr = st.slider(
                "snr",
                min_value=0.05,
                max_value=0.25,
                value=0.16,
                step=0.01,
                help="signal to noise ratio"
            )
            denoise = st.checkbox("denoise", value=True)
            num_samples = st.slider(
                "number of samples", min_value=1, max_value=16, step=1, value=1
            )
            # continuous = st.radio("continuous", (True, False))
            animate = st.checkbox("animate", value=True)
            anim_info = st.empty()
            if animate:
                if num_samples != 1:
                    anim_info = st.info(
                        "Animation will be done for a single sample only"
                    )
                    num_samples = 1

                update_every = st.number_input(
                    "Animation frame update frequency",
                    min_value=1,
                    max_value=model.sde.N,
                    value=10,
                )
                save_mp4 = st.checkbox(
                    "save as .mp4 file (fps=25)", value=False
                )

        # ODE-sampling configuration
        elif sampling_method == "ODE":

            ode_method = st.selectbox(
                "solver method",
                ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
            )
            ode_rtol = st.number_input(
                "rtol",
                min_value=0.,
                max_value=0.1,
                value=1e-5,
                step=1e-6,
                format="%.6f",
                help="relative tolerance",
            )
            ode_atol = st.number_input(
                "atol",
                min_value=0.,
                max_value=0.1,
                value=1e-5,
                step=1e-6,
                format="%.6f",
                help="absolute tolerance",
            )
            denoise = st.checkbox("denoise", value=True)
            num_samples = st.slider(
                "number of samples", min_value=1, max_value=16, step=1, value=1
            )

        # Latent Gaussian sampling configuration
        else:
            latent_std = st.slider(
                "Latent std", min_value=0., max_value=25., value=5., step=0.1
            )

        button_sample = st.form_submit_button("Sample")

    # Sampling button action
    if button_sample:
        # For an accurate timing, would need to time within the actual
        # sampling loops.
        sample_start = time.time()

        if sampling_method == "PC (predictor-corrector)":
            anim_info.empty()
            if animate:
                col_right.subheader("Sample")
                output = col_right.empty()
                if save_mp4:
                    import imageio
                    outvid = "sampling.mp4"
                    writer = imageio.get_writer(outvid, fps=25)

            for i, sample in enumerate(
                stqdm(
                    model._sampling_generator(
                        last_only=not animate,
                        sampling_method="pc",
                        num_samples=num_samples,
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
                    st_container=col_right,
                    frontend=True,
                    backend=True,
                )
            ):
                if animate and i * n_steps_each % update_every == 0:
                    output.image(
                        bchw_to_st(sample), clamp=True, output_format="PNG"
                    )
                    if save_mp4:
                        writer.append_data(
                            (bchw_to_st(sample)[0]*255).clip(0, 255).astype(
                                np.uint8
                            )
                        )
                else:
                    # Iterate all the way to the end to get the final sample
                    pass

            if animate and save_mp4:
                writer.close()
                st.session_state.sampling_video = outvid

            st.session_state.sample = sample
            # Skip showing the final sample later again, because it's already
            # shown when animating the sampling.
            st.session_state.skip_show_sample = True

        elif sampling_method == "ODE":
            with col_right.spinner("Solving the reverse ODE ..."):
                st.session_state.sample = model.sample(
                    sampling_method="ode",
                    num_samples=num_samples,
                    noise_removal=denoise,
                    ode_method=ode_method,
                    ode_rtol=ode_rtol,
                    ode_atol=ode_atol,
                )

        else:
            z = (torch.randn((1, 16, 16, 16)) * latent_std).to(model.device)
            st.session_state.sample = model.ae_model.decode(z)

        sample_end = time.time()
        st.session_state.sampling_time = sample_end - sample_start

    # Display current sample, sampling time, sampling video
    if (
        st.session_state.get("sample", None) is not None
        and not st.session_state.get("skip_show_sample", False)
    ):
        col_right.subheader("Sample")
        col_right.image(
            bchw_to_st(st.session_state.sample),
            clamp=True,
            output_format="PNG"
        )
    else:
        st.session_state.skip_show_sample = False

    if st.session_state.get("sampling_time", None) is not None:
        col_right.write(
            f"Sampling time: < {st.session_state.sampling_time:.1f} s"
        )

    if st.session_state.get("sampling_video", None) is not None:
        col_right.subheader("Sampling video")
        col_right.video(st.session_state.sampling_video)


def clear_images():
    if "input_image" in st.session_state:
        st.session_state.input_image = None
    if "rec_image" in st.session_state:
        st.session_state.rec_image = None
    if "sample" in st.session_state:
        st.session_state.sample = None
    if "sampling_time" in st.session_state:
        st.session_state.sampling_time = None
    if "sampling_video" in st.session_state:
        st.session_state.sampling_video = None


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # .. Load and resolve model configuration .................................
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

    # .. Start of the Streamlit script ........................................
    st.set_page_config(layout="wide", page_icon=":game_die:")
    st.title("Score-based generative modelling in latent space")
    if ckpt is not None:
        st.markdown(f"_checkpoint: {ckpt}_")

    # Load the model and data
    gpu = True
    eval_mode = True
    model, global_step = load_model(config.model, ckpt, gpu, eval_mode)
    dsets = get_data(config.data).datasets

    st.markdown(f"_global step: {global_step}_")
    st.button("⚠️ Clear images", on_click=clear_images)
    st.markdown("____")

    # Part 1: autoencoder reconstruction
    run_reconstruction(model, dsets)

    # Part 2: sampling
    run_sampling(model)

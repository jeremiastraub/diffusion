"""Evaluation script for sample generation.
"""
import argparse, os, sys, datetime, glob, logging

import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from main import instantiate_from_config, resolve_based_on
from tqdm import tqdm

# -----------------------------------------------------------------------------


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        const=True,
        default=8,
        nargs="?",
        help="number of samples",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        const=True,
        default=16,
        nargs="?",
        help="batch size for sampling",
    )
    parser.add_argument(
        "-progr",
        "--progression",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Whether to make a progression plot",
    )
    parser.add_argument(
        "-out",
        "--model_output",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Additionally display the direct model output",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right.",
        default=list(),
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="custom output directory"
    )
    parser.add_argument(
        "-sf",
        "--single_file",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="Whether to store all samples in a single grid plot",
    )
    parser.add_argument(
        "-dpi",
        "--dpi",
        type=int,
        const=True,
        default=256,
        nargs="?",
        help="figure DPI",
    )
    return parser


def tensor_to_imshow(x):
    """Expects input of shape (B, C, H, W)"""
    x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    x = 0.5 * (x + 1.)
    return x.clip(0., 1.)


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
        print("Deleted the restore-ckpt path from the config.")

    model = instantiate_from_config(config)

    if pl_sd["state_dict"] is not None:
        missing, unexpected = model.load_state_dict(
            pl_sd["state_dict"], strict=False
        )
        if missing:
            logging.debug(f"Missing Keys in State Dict: {missing}")
        if unexpected:
            logging.debug(f"Unexpected Keys in State Dict: {unexpected}")

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()

    return model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    logdir = None
    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = os.path.splitext(opt.resume)[0].split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
                logdir = "/".join(paths[:idx])
                base_configs = sorted(
                    glob.glob(os.path.join(logdir, "configs/*-project.yaml"))
                )
            except ValueError:
                assert "models" in paths, paths
                idx = len(paths)-paths[::-1].index("models")
                logdir = "logs/" + "_".join(paths[idx:])
                base_configs = glob.glob(
                    os.path.join(*paths[:-1], "*-project.yaml")
                )
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
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    config = resolve_based_on(config)

    # Set up logdir
    if opt.savedir:
        logdir = opt.savedir
    elif not opt.resume:
        raise ValueError("If not resuming from checkpoint, provide a savedir")
    else:
        logdir = os.path.join(logdir, "eval")

    os.makedirs(logdir, exist_ok=True)

    # Load model
    gpu = True
    eval_mode = True
    model, global_step = load_model(config.model, ckpt, gpu, eval_mode)

    cond_key = model.cond_key if model.diff_model.class_conditional else None

    workdir = os.path.join(logdir, f"{global_step}")
    os.makedirs(workdir, exist_ok=True)

    n_samples = opt.n_samples
    n_batches = int(np.ceil(n_samples / opt.batch_size))
    n_last_batch = n_samples % opt.batch_size

    # Sample from representation prior and decode it
    repr_cond = None
    if model.repr_ae is not None and not model.repr_ae.decoder.time_conditional:
        repr = torch.randn(
            size=(n_samples,) + tuple(model.repr_ae.latent_shape),
            device=model.device,
        )
        repr = torch.split(repr, opt.batch_size)

        # TODO repr_ae decoder kwargs not possible

        repr_cond = [model.repr_ae.decode(r) for r in repr]

    # Sample from class prior distribution
    classes = None
    if model.diff_model.class_conditional:
        classes = torch.tensor(
            np.random.randint(model.diff_model.num_classes, size=n_samples),
            device=model.device,
        )
    y = torch.split(classes, opt.batch_size) if classes is not None else None

    title_info = []

    if model.repr_ae is not None:
        title_info.append(f"KL-w = {model.repr_kl_weight}")
        # title_info.append(f"d_r = {model.repr_ae.embed_dim}")


    if not opt.progression:
        # Compute samples
        samples = []
        for batch_idx in tqdm(range(n_batches), desc="Sampling batches"):
            model_kwargs = {}

            if repr_cond is not None:
                model_kwargs["repr"] =  repr_cond[batch_idx]
            if y is not None:
                model_kwargs["y"] = y[batch_idx]

            num_samples = opt.batch_size
            if n_last_batch and batch_idx == n_batches - 1:
                num_samples = n_last_batch

            sample = model.sample(
                num_samples=num_samples, model_kwargs=model_kwargs,
            )
            samples.append(tensor_to_imshow(sample))

        samples = np.concatenate(samples)
        assert len(samples) == n_samples

        if opt.single_file:
            nrows = int(np.sqrt(n_samples))
            ncols = nrows + int(np.ceil((n_samples - nrows**2) / nrows))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
            fig.suptitle(f"Samples: {', '.join(title_info)}")
            for i, img in enumerate(samples):
                axes[i].imshow(img, cmap="binary_r")
                axes[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            fig.savefig(
                os.path.join(workdir, f"{timestamp}_samples.pdf"),
                dpi=opt.dpi,
            )
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            out_dir = os.path.join(workdir, f"{timestamp}_samples")
            os.makedirs(out_dir, exist_ok=False)

            for i, img in enumerate(samples):
                fig, ax = plt.subplots()
                ax.imshow(img, cmap="binary_r")
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                ax.set_frame_on(False)
                fig.savefig(
                    os.path.join(out_dir, f"{i}.pdf"),
                    # dpi=opt.dpi,
                    dpi="figure",
                    bbox_inches="tight",
                    pad_inches=0.,
                )
                plt.close(fig)

    else:
        n_progr_steps = 10
        progr_steps = np.linspace(0, model.sde.N-1, n_progr_steps, dtype=int)

        sample_progression = []
        estimate_progression = []

        if n_batches > 1:
            # Have to combine progressions differently
            raise NotImplementedError()

        # Sampling
        for i in tqdm(range(n_batches), desc="Sampling"):
            model_kwargs = {}
            if classes is not None:
                model_kwargs["y"] = classes[i]

            for n, (sample_step, estimate) in enumerate(
                model.sampling_generator(
                    denoise=False, last_only=False, num_samples=n_samples,
                )
            ):
                if n in progr_steps:
                    sample_progression.append(tensor_to_imshow(sample_step))
                    estimate_progression.append(
                        tensor_to_imshow(model.decode(estimate))
                    )

            progr_final = tensor_to_imshow(sample_step)

        sample_progression = np.concatenate(sample_progression)
        estimate_progression = np.concatenate(estimate_progression)

        # Save Images ---------------------------------------------------------

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = os.path.join(workdir, f"{timestamp}_sampling_progr")
        os.makedirs(out_dir, exist_ok=False)

        for i, img in enumerate(progr_final):
            fig, ax = plt.subplots()
            ax.imshow(img, cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"final{i}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for j, img in enumerate(sample_progression):
            i = j % opt.n_samples
            j = j // opt.n_samples

            fig, ax = plt.subplots()
            ax.imshow(img, cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"sampling{i}-{j}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for j, img in enumerate(estimate_progression):
            i = j % opt.n_samples
            j = j // opt.n_samples

            fig, ax = plt.subplots()
            ax.imshow(img, cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"estimate{i}-{j}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

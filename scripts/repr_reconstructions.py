"""Visualize image 'reconstructions' in a representation learning setup.
"""
import numpy as np
import torch

import argparse, os, sys, datetime, glob

from omegaconf import OmegaConf
from main import instantiate_from_config, resolve_based_on
from tqdm import tqdm

import matplotlib.pyplot as plt

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
        "-d",
        "--data",
        type=str,
        default="",
        nargs="?",
        help="dataset",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="validation",
        nargs="?",
        help="dataset split; can be: train, validation",
    )
    parser.add_argument(
        "-st",
        "--style",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help=(
            "Whether to use only the style information for reconstructions "
            "(only for class-conditional setup)"
        ),
    )
    parser.add_argument(
        "-ni",
        "--n_inputs",
        type=int,
        const=True,
        default=8,
        nargs="?",
        help="Number of different input images",
    )
    parser.add_argument(
        "-nr",
        "--n_reconstructions",
        type=int,
        const=True,
        default=7,
        nargs="?",
        help="Number of reconstructions per input image",
    )
    parser.add_argument(
        "-progr",
        "--progression",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Whether to make a sampling progression plot",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        const=True,
        default=64,
        nargs="?",
        help="batch size for sampling",
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
    x = x.detach().cpu().numpy().transpose(1, 2, 0)
    x = 0.5 * (x + 1.)
    return x.clip(0., 1.)


def get_data(config):
    data = instantiate_from_config(config)
    data.prepare_data()
    data.setup()
    return data


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
            print(f"Missing Keys in State Dict: {missing}")
        if unexpected:
            print(f"Unexpected Keys in State Dict: {unexpected}")

    if eval_mode:
        model.eval()
    if gpu:
        model.cuda()

    return model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.progression:
        assert opt.n_reconstructions == 1

    logdir = None
    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
                logdir = "/".join(paths[:idx])
                base_configs = sorted(
                    glob.glob(os.path.join(logdir, "configs/*-project.yaml"))
                )
            except ValueError:
                assert "models" in paths, paths
                idx = len(paths)-paths[::-1].index("models")+1
                logdir = "logs/" + "/".join(paths[idx:])
                base_configs = glob.glob(
                    os.path.join(*paths[:-1], "/*-project.yaml")
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

    if opt.data:
        config.based_on = {**config.get("based_on", {}), **{"data": opt.data}}
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

    # Get the data
    data = get_data(config.data)
    dset = data.datasets[opt.split]
    data_name = dset.__class__.__name__

    workdir = os.path.join(logdir, data_name, f"{global_step}")
    os.makedirs(workdir, exist_ok=True)


    indices = np.random.choice(len(dset), size=opt.n_inputs, replace=False)
    # indices = np.array([4141, 1465, 1805, 2749, 1317, 2716, 3185, 972, 2916, 1891])

    input_images = []
    input_batches = []
    input_classes = []

    # Get input images and class labels
    for idx in indices:
        batch = dset[idx]
        image, y = model.get_input(
            batch,
            image_key="image",
            cond_key=cond_key,
            device=model.device,
            encode=False,
        )
        input_images.append(image)
        if y is not None:
            input_classes.append(torch.tensor([y], device=model.device))

    # Get (latent) Diffusion-Model input batches
    # if model.ae_model is not None:
    #     for idx in indices:
    #         batch = dset[idx]
    #         b, _ = model.get_input(
    #             batch,
    #             image_key="image",
    #             cond_key=cond_key,
    #             device=model.device,
    #             encode=True,
    #         )
    #         input_batches.append(b)
    # else:
    #     input_batches = input_images
    for idx in indices:
        batch = dset[idx]
        b, _ = model.get_input(
            batch,
            image_key="image",
            cond_key=cond_key,
            device=model.device,
            encode=True,
        )
        input_batches.append(b)

    # Prepare sampling conditioning information
    print("Preparing conditioning information...")
    input_images = torch.cat(input_images, dim=0)
    input_batches = torch.cat(input_batches, dim=0)
    input_classes = torch.cat(input_classes, dim=0) if input_classes else None


    input_batches = input_batches.repeat(
        opt.n_reconstructions, *([1]*(input_batches.ndim - 1))
    )
    input_batches = torch.split(input_batches, opt.batch_size)



    classes = None
    if input_classes is not None:
        if not opt.style:
            # Use input classes
            classes = input_classes.repeat(opt.n_reconstructions)
        else:
            # Choose new (random) class indices for style reconstructions
            # classes_arr = np.random.choice(
            #     model.diff_model.num_classes,
            #     size=opt.n_reconstructions,
            #     replace=False,
            # )
            classes_arr = np.arange(opt.n_reconstructions)
            classes = torch.tensor(classes_arr, device=model.device)
            classes = classes.repeat_interleave(opt.n_inputs, dim=0)

        classes = torch.split(classes, opt.batch_size)


    reconstructions = []
    num_rec = opt.n_inputs * opt.n_reconstructions
    num_left = num_rec
    num_steps = int(np.ceil(num_rec / opt.batch_size))

    if not opt.progression:

        # Compute reconstructions
        with torch.no_grad(), model.ema_scope():

            sampler, _ = model.prepare_sampling()
            # sampler, _ = model.prepare_sampling(predictor="ddim_x0_sampling")

            for i in tqdm(
                range(num_steps), desc="Reconstructing", total=num_steps
            ):
                model_kwargs = {}
                if classes is not None:
                    model_kwargs["y"] = classes[i]

                num_samples = min(opt.batch_size, num_left, num_rec)

                z = sampler.reconstruct(input_batches[i], **model_kwargs)
                # z = sampler.reconstruct(input_batches[i], use_r_above=0.7, **model_kwargs)
                x = model.decode(z)

                reconstructions.append(x)
                num_left -= num_samples

    else:
        n_progr_steps = 10
        progr_steps = np.linspace(0, model.sde.N-1, n_progr_steps, dtype=int)

        sample_progression = []
        estimate_progression = []

        if num_steps > 1:
            # Have to combine progressions differently
            raise NotImplementedError()

        # Compute reconstructions
        with torch.no_grad(), model.ema_scope():

            sampler, _ = model.prepare_sampling()
            sampler.denoise = False

            for i in tqdm(
                range(num_steps), desc="Reconstructing", total=num_steps
            ):
                model_kwargs = {}
                if classes is not None:
                    model_kwargs["y"] = classes[i]

                for n, (sample_step, estimate) in enumerate(
                    sampler.reconstruction_progression(
                        input_batches[i], **model_kwargs
                    )
                ):
                    if n in progr_steps:
                        sample_progression.append(model.decode(sample_step))
                        estimate_progression.append(model.decode(estimate))

                progr_reconstruction = model.decode(sample_step)

        sample_progression = torch.cat(sample_progression, dim=0)
        estimate_progression = torch.cat(estimate_progression, dim=0)

    # -------------------------------------------------------------------------
    # Save images

    if opt.single_file:
        if opt.progression:
            raise NotImplementedError()

        reconstructions = torch.cat(reconstructions, dim=0)
        img_grid = torch.cat([input_images, reconstructions])
        assert len(img_grid) == opt.n_inputs * (opt.n_reconstructions + 1)

        fig, axes = plt.subplots(nrows=opt.n_reconstructions+1, ncols=opt.n_inputs)
        axes = axes.ravel()
        fig.suptitle(
            f"{'Style ' if opt.style else ''}Reconstructions: "
            # f"r : {model.repr_ae.embed_dim} x {} x {},"
            f"KL-w = {model.repr_kl_weight}"
        )
        for i, img in enumerate(img_grid):
            axes[i].imshow(tensor_to_imshow(img), cmap="binary_r")
            axes[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if i < opt.n_inputs:
                axes[i].set_title(f"im{i}")
            if opt.style and i % opt.n_inputs == 0 and i != 0:
                axes[i].set_ylabel(f"{classes_arr[i//opt.n_inputs-1]}")

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig_name = "rec_style" if opt.style else "reconstructions"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        fig.savefig(
            os.path.join(workdir, f"{timestamp}_{fig_name}.pdf"),
            dpi=opt.dpi,
        )

    elif not opt.progression:
        reconstructions = torch.cat(reconstructions, dim=0)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if opt.style:
            out_dir = os.path.join(workdir, f"{timestamp}_rec_style")
        else:
            out_dir = os.path.join(workdir, f"{timestamp}_reconstructions")
        os.makedirs(out_dir, exist_ok=False)

        for i, input_img in enumerate(input_images):
            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(input_img), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"input{i}-{indices[i]}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for j, rec in enumerate(reconstructions):
            i = j % opt.n_inputs
            j = j // opt.n_inputs

            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(rec), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"rec{i}-{j}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = os.path.join(workdir, f"{timestamp}_rec_progr")
        os.makedirs(out_dir, exist_ok=False)

        for i, input_img in enumerate(input_images):
            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(input_img), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"input{i}-{indices[i]}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for i, rec in enumerate(progr_reconstruction):
            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(rec), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"reconstruction{i}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for j, rec in enumerate(sample_progression):
            i = j % opt.n_inputs
            j = j // opt.n_inputs

            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(rec), cmap="binary_r")
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

        for j, rec in enumerate(estimate_progression):
            i = j % opt.n_inputs
            j = j // opt.n_inputs

            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(rec), cmap="binary_r")
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

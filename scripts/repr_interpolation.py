"""Interpolate in representation space.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse, os, sys, datetime, glob, logging

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
        "-rd",
        "--random",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Sample representations from prior",
    )
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
        "-m",
        "--mode",
        type=str,
        default="slerp",
        nargs="?",
        help="interpolation mode",
    )
    parser.add_argument(
        "-n",
        "--n_interpolations",
        type=int,
        const=True,
        default=4,
        nargs="?",
        help="Number of interpolations",
    )
    parser.add_argument(
        "-ni",
        "--n_intermediate",
        type=int,
        const=True,
        default=6,
        nargs="?",
        help="Number of intermediate interpolation steps",
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
            logging.debug(f"Missing Keys in State Dict: {missing}")
        if unexpected:
            logging.debug(f"Unexpected Keys in State Dict: {unexpected}")

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()

    return model, global_step


def lerp_interpolation(low, high, n):
    weights = torch.linspace(0., 1., n+2, device=low.device)[1:-1]
    intermediate = torch.stack(
        [torch.lerp(low, high, w) for w in weights]
    )
    return intermediate

def slerp_interpolation(low, high, n):
    # low, high: N x C x H x W
    weights = torch.linspace(0., 1., n+2, device=low.device)[1:-1]
    low_norm = low / torch.norm(
        low.reshape(low.shape[0], -1), dim=1
    )[:, None, None, None]
    high_norm = high / torch.norm(
        high.reshape(high.shape[0], -1), dim=1
    )[:, None, None, None]
    omega = torch.acos((low_norm * high_norm).sum(dim=[1, 2, 3]))
    so = torch.sin(omega)
    intermediate = torch.stack(
        [
            (
                (torch.sin((1. - w) * omega)/so)[:, None, None, None] * low
                + (torch.sin(w * omega)/so)[:, None, None, None] * high
            ) for w in weights
        ]
    )
    return intermediate


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

    classes = None
    classes_inter = None
    images_start = None
    images_end = None
    repr_start = None
    repr_end = None
    repr_inter = None
    repr_inter_seq = None

    """
    if not opt.random:
        get input image pairs of the same class
        get repr pairs from input images
    else:
        sample repr pairs and classes
        sample "input" images (combine with interpolation sampling?)
        
    sample intermediate steps
    arrange them between start and ending image
    """

    if not opt.random:
        input_images = []

        # Get input images and class labels
        if model.diff_model.class_conditional:
            # raise NotImplementedError(
            #     "Need to find 'target' indices of the same class"
            # )
            indices_origin = np.random.choice(
                len(dset), size=opt.n_interpolations, replace=False
            )
            target_indices = []
            target_images = []
            classes = []
            for idx in indices_origin:
                batch = dset[idx]
                image, y = model.get_input(
                    batch, image_key="image", cond_key=cond_key, encode=False
                )
                assert y is not None
                input_images.append(image)
                classes.append(torch.tensor([y], device=model.device))

                target_y = None
                while target_y != y:
                    target_idx = np.random.choice(len(dset))
                    target_image, target_y = model.get_input(
                        dset[target_idx],
                        image_key="image",
                        cond_key=cond_key,
                        encode=False
                    )

                # target_idx = np.random.choice(len(dset))
                # target_image, target_y = model.get_input(
                #     dset[target_idx],
                #     image_key="image",
                #     cond_key=cond_key,
                #     encode=False
                # )

                target_images.append(target_image)
                target_indices.append(target_idx)

            classes = torch.cat(classes, dim=0)
            input_images += target_images
            input_images = torch.cat(input_images, dim=0)
            indices = list(indices_origin) + target_indices

        else:
            indices = np.random.choice(
                len(dset), size=2*opt.n_interpolations, replace=False
            )
            for idx in indices:
                batch = dset[idx]
                image, _ = model.get_input(
                    batch, image_key="image", cond_key=cond_key, encode=False
                )
                input_images.append(image)

            input_images = torch.cat(input_images, dim=0)

        # Get representations
        input_batches = []
        for idx in indices:
            batch = dset[idx]
            b, _ = model.get_input(
                batch, image_key="image", cond_key=cond_key, encode=True
            )
            input_batches.append(b)

        input_batches = torch.cat(input_batches, dim=0)

        enc_kwargs = {}
        if model.repr_ae.encoder.class_conditional:
            enc_kwargs["y"] = classes.repeat(2)
        if model.repr_ae.encoder.time_conditional:
            raise NotImplementedError()
        with torch.no_grad():
            # repr = model.repr_ae.encode(input_batches, **enc_kwargs).sample()
            repr = model.repr_ae.encode(input_batches, **enc_kwargs).mode()

        repr_start, repr_end = torch.chunk(repr, 2)
        images_start, images_end = torch.chunk(input_images, 2)

    else:
        model_kwargs = {}
        if model.diff_model.class_conditional:
            classes = torch.tensor(
                np.random.randint(
                    model.diff_model.num_classes, size=opt.n_interpolations
                ),
                device=model.device,
            )
            model_kwargs["y"] = classes.repeat(2)

        repr = torch.randn(
            size=(2*opt.n_interpolations,) + tuple(model.repr_ae.latent_shape),
            device=model.device,
        )
        print("Sampling start and ending point images...")
        with torch.no_grad():
            dec_kwargs = {}
            if model.repr_ae.decoder.class_conditional:
                dec_kwargs["y"] = classes.repeat(2)
            model_kwargs["repr"] = model.repr_ae.decode(repr, **dec_kwargs)
            images = model.sample(
                num_samples=2*opt.n_interpolations, model_kwargs=model_kwargs
            )

        repr_start, repr_end = torch.chunk(repr, 2)
        images_start, images_end = torch.chunk(images, 2)

    print("Preparing conditioning information...")
    # NOTE compute representation interpolation; transpose first two dims to
    #      get row-wise tensor, then flatten to grid (via reshape)
    if opt.mode == "lerp" or opt.mode == "linear":
        repr_inter = lerp_interpolation(
            repr_start, repr_end, n=opt.n_intermediate
        ).transpose(0, 1).reshape(-1, *repr_start.shape[1:])
    elif opt.mode == "slerp" or opt.mode == "spherical-linear":
        repr_inter = slerp_interpolation(
            repr_start, repr_end, n=opt.n_intermediate
        ).transpose(0, 1).reshape(-1, *repr_start.shape[1:])
    else:
        raise ValueError("Invalid interpolation mode")

    repr_inter = torch.split(repr_inter, opt.batch_size, dim=0)

    if classes is not None:
        classes_inter = torch.repeat_interleave(classes, opt.n_intermediate)
        classes_inter = torch.split(classes_inter, opt.batch_size, dim=0)

    if model.repr_ae.decoder.time_conditional:
        raise NotImplementedError()
    with torch.no_grad():
        if classes_inter is None:
            repr_cond = [model.repr_ae.decode(r) for r in repr_inter]
        else:
            repr_cond = [
                model.repr_ae.decode(r, y=c)
                for r, c in zip(repr_inter, classes_inter)
            ]

    interpolations = []
    num_samples = opt.n_interpolations * opt.n_intermediate
    num_left = num_samples
    num_steps = int(np.ceil(num_samples / opt.batch_size))

    # Size checks
    for q in [classes_inter, repr_inter, repr_inter_seq]:
        if q is not None:
            assert len(q) == num_steps

    # Sample interpolations
    for i in tqdm(range(num_steps), desc="Interpolating", total=num_steps):
        model_kwargs = {}
        if classes_inter is not None:
            model_kwargs["y"] = classes_inter[i]
        if repr_cond is not None:
            model_kwargs["repr"] = repr_cond[i]
        if repr_inter_seq is not None:
            model_kwargs["repr_seq"] = repr_inter_seq[i]

        num_samples=min(opt.batch_size, num_left, num_samples)
        x_int = model.sample(
            num_samples=num_samples,
            model_kwargs=model_kwargs,
        )
        interpolations.append(x_int)
        num_left -= num_samples

    interpolations = torch.cat(interpolations, dim=0)


    if opt.single_file:
        # Again, combine to row-wise tensor first
        img_grid = torch.cat(
            [
                images_start.unsqueeze(1),
                interpolations.reshape(
                    opt.n_interpolations,
                    opt.n_intermediate,
                    *interpolations.shape[1:]
                ),
                images_end.unsqueeze(1),
            ],
            dim=1,
        ).reshape(-1, *interpolations.shape[1:])

        assert len(img_grid) == opt.n_interpolations * (opt.n_intermediate + 2)

        fig, axes = plt.subplots(
            nrows=opt.n_interpolations, ncols=opt.n_intermediate+2
        )
        axes = axes.ravel()
        fig.suptitle(f"{opt.mode} interpolations")
        for i, img in enumerate(img_grid):
            axes[i].imshow(tensor_to_imshow(img), cmap="binary_r")
            axes[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        fig_name = f"{opt.mode}-interpolations"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        fig.savefig(
            os.path.join(workdir, f"{timestamp}_{fig_name}.pdf"),
            dpi=opt.dpi,
        )

    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = os.path.join(workdir, f"{timestamp}_{opt.mode}_interpolations")
        os.makedirs(out_dir, exist_ok=False)

        for i, (start, end) in enumerate(zip(images_start, images_end)):
            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(start), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                # os.path.join(out_dir, f"s{i}.pdf"),
                os.path.join(out_dir, f"s{i}-{indices[i]}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(end), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                # os.path.join(out_dir, f"t{i}.pdf"),
                os.path.join(out_dir, f"t{i}-{indices[i+opt.n_interpolations]}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

        for i, sample in enumerate(interpolations):
            j = i % opt.n_intermediate
            i = i // opt.n_intermediate

            fig, ax = plt.subplots()
            ax.imshow(tensor_to_imshow(sample), cmap="binary_r")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_frame_on(False)
            fig.savefig(
                os.path.join(out_dir, f"int{i}-{j}.pdf"),
                # dpi=opt.dpi,
                dpi="figure",
                bbox_inches="tight",
                pad_inches=0.,
            )
            plt.close(fig)

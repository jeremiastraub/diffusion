import argparse, os, sys, datetime, glob, importlib, csv, time

from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch_fidelity import calculate_metrics

# -----------------------------------------------------------------------------

# root directory for base configurations
BASE_CONFIG_PATH = "configs/base_configs"


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
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
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of "
        "the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="",
        help="directory for logging dat shit",
    )

    return parser


def resolve_based_on(cfg, root: str=BASE_CONFIG_PATH):
    """Resolve any based_on entries in the configuration by updating the base
    configuration recursively. The updating is done via OmegaConf.merge.

    The ``based_on`` entry in ``cfg`` can be
        - str: "<path>.<key1>.<key...>"; <path> specifies the path the to base
            configuration file relative to `root` (_without_ the .yaml file
            extension). Append keys to select specific entries within that cfg.
        - list[str]: list of such strings. They will be parsed sequentially.
        - dict (recommended): target path strings keyed by a base-key.
            Prepends the base-keys as top-level keys to the target base
            configurations. base-keys may contain multiple keys separated by
            a ".".
            For example
            ```
            based_on:
              foo.bar: config_file.some_key
            ```
            will be translated into
            ```
            foo:
              bar: <config_file.some_key-dictionary>
            ```
    """
    def _resolve_single_ref(cfg, based_on, *, base_key=None):
        split = based_on.split(".", 1)
        if len(split) == 1:
            file, keys = split[0], []
        else:
            file, keys = split
            keys = keys.split(".")

        base_path = os.path.join(root, file) + ".yaml"
        base = OmegaConf.load(base_path)

        for k in keys:
            base = base[k]

        if "based_on" in base:
            raise ValueError(
                "Found 'based_on' entry in base configuration. Recursive use "
                "of 'based_on' in this way is not supported."
            )

        if base_key is not None:
            base_keys = base_key.split(".")

            def nested(keys):
                """Turn list of keys into nested dict"""
                return {keys.pop(0): nested(keys)} if keys else {**base}

            return OmegaConf.merge(nested(base_keys), cfg)
        else:
            return OmegaConf.merge(base, cfg)

    based_on = cfg.pop("based_on", None)

    if based_on is None:
        return cfg

    elif isinstance(based_on, str):
        return _resolve_single_ref(cfg, based_on)

    elif isinstance(based_on, list):
        updated_cfg = cfg.copy()
        for b in based_on:
            updated_cfg = _resolve_single_ref(updated_cfg, b)
        return updated_cfg

    else:
        # dict-like based_on
        updated_cfg = cfg.copy()
        for k, b in based_on.items():
            updated_cfg = _resolve_single_ref(updated_cfg, b, base_key=k)
        return updated_cfg


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(
        k for k in vars(args)
        if getattr(opt, k) != getattr(args, k)
        or type(getattr(opt, k)) != type(getattr(args, k))
    )


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a
    pytorch dataset.
    """
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = (
            num_workers if num_workers is not None else batch_size*2
        )
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


class SetupCallback(Callback):
    def __init__(
        self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now))
            )

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now))
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        *,
        batch_frequency,
        max_images,
        min_steps=0,
        clamp=True,
        rescale=True,
        use_exponential_steps=False,
        log_on_batch_idx=False,
        **log_kwargs
    ):
        """Callback for logging images.

        .. note::

            Requires the ``log_images`` method of the respective pl-module
            to return a dict containing the images keyed by logging key.

        Args:
            batch_frequency: Number of batches between logging steps.
            max_images: Maximum number of logged images. Set to zero to disable
                image logging.
            min_steps: Minimum number of steps before logging starts
            clamp: Whether to clamp image to [0, 1]
            rescale: Whether to transform image data from [-1, 1] to [0, 1]
            use_exponential_steps: If True, logging steps are powers of 2;
                Ignores ``batch_frequency`` if logging is not disabled.
            log_on_batch_idx: Whether to log based on the batch index of the
                current epoch. If False, logging is based on the global step.
            **log_kwargs: Passed to pl_module.log_images
        """
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.min_steps = min_steps
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.use_exponential_steps = use_exponential_steps
        self.current_exponent = None
        self.executed_steps = list()
        self.clamp = clamp
        self.log_on_batch_idx = log_on_batch_idx
        self.log_kwargs = log_kwargs

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.) / 2. # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.) / 2. # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = (
            batch_idx if self.log_on_batch_idx else pl_module.global_step
        )
        # Initialize exponent on first ``log_img`` call
        if self.current_exponent is None:
            if self.log_on_batch_idx:
                self.current_exponent = 0
            else:
                self.current_exponent = (
                    0 if pl_module.global_step == 0
                    else int(np.log2(pl_module.global_step)) + 1
                )
        if (
            self.max_images > 0
            and self.check_frequency(check_idx)
            # only one logging step per global-step when using
            # accumulate_grad_batches > 1
            and pl_module.global_step not in self.executed_steps
        ):
            self.executed_steps.append(pl_module.global_step)
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx
            )

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, n):
        if self.use_exponential_steps and n == 2 ** self.current_exponent:
            self.current_exponent += 1
            if n >= self.min_steps:
                return True
        elif n % self.batch_freq == 0 and n >= self.min_steps:
            return True
        return False

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="val")


class FIDelity(Callback):
    def __init__(
        self,
        *,
        data_cfg,
        split="validation",
        num_images=5000,
        isc=True,
        kid=True,
        fid=True,
        epoch_frequency=1,
        step_frequency=None,
        min_epochs=1,
        min_steps=0,
        input_key="inputs",
        output_key="samples",
        load_input_from=None,
        save_input_to=None,
        save_output_to=None,
        clamp=True,
        keep_intermediate_output=False,
        log_images_kwargs=None,
        **fid_kwargs
    ):
        """Callback for evaluating and logging FID, IS, etc.
        Based on https://torch-fidelity.readthedocs.io/en/latest/api.html.

        .. note::

            Requires the ``log_images`` method of the respective pl-module
            to return a dict containing the ``input_key`` and ``output_key``
            keys (these are passed to the logging method as ``to_log``).

        Args:
            data_cfg: cutlit.DataModuleFromConfig configuration. Passed to
                cutlit.instantiate_from_config.
            split: dset split to use, can be one of: train, validation, test.
            num_images: Number of images contained in the dataset configured by
                ``data_cfg``. If < 0, the whole dataset split is used.
                Note that the effective number of created images depends on the
                number of images returned by the pl_module.log_images method.
            isc: Whether to calculate the Inception Score
            kid: Whether to calculate the Kernel Inception Distance
            fid: Whether to calculate the Frechet Inception Distance
            epoch_frequency: Number of epochs between score evaluations. Set to
                None to disable epoch-periodic evaluation.
            step_frequency: Number of steps between score evaluations. Set to
                None to disable step-periodic evaluation.
            min_epochs: If epoch-periodic evaluation is enabled, defines
                starting threshold.
            min_steps: If step-periodic evaluation is enabled, defines starting
                threshold.
            input_key: Input image logging key
            output_key: Output image logging key
            load_input_from: Custom path to directory containing the input
                images (e.g. previously written there via save_input_to).
            save_input_to: Custom path to directory where the input images are
                written to. May not be given together with load_input_from.
            save_output_to: Custom path to directory where the output images
                are written to.
            clamp: Whether to clamp images to [0, 1]
            log_images_kwargs: Passed to pl_module.log_images
            keep_intermediate_output: Whether to store output images for each
                evaluation separately. If False, overwrites previous outputs.
            **fid_kwargs: Passed to torch_fidelity.calculate_metrics
        """
        super().__init__()
        self.data_cfg = data_cfg
        self.split = split
        self.num_images = num_images
        self.input_key = input_key
        self.output_key = output_key
        self.epoch_frequency = epoch_frequency
        self.step_frequency = step_frequency
        self.min_epochs = min_epochs
        self.min_steps = min_steps
        assert not (load_input_from is not None and save_input_to is not None)
        self.load_input_from = load_input_from
        self.save_input_to = save_input_to
        self.save_output_to = save_output_to
        self.keep_intermediate = keep_intermediate_output

        self.isc = isc
        self.kid = kid
        self.fid = fid
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs or {}
        self.fid_kwargs = fid_kwargs

        self.prepared = False
        self.input_cached = False
        self.executed_steps = list()

    @rank_zero_only
    def prepare(self, logdir):
        if not self.prepared:
            self.init_data()
            self._init_folders(logdir)
            self.prepared = True

    @rank_zero_only
    def _init_folders(self, logdir):
        # set up directories where the images will be stored at
        workdir = os.path.join(logdir, "fidelity", self.dset_name)
        indir = os.path.join(workdir, self.input_key)
        outdir = os.path.join(workdir, self.output_key)

        if self.load_input_from is not None:
            indir = self.load_input_from
            if not os.path.isdir(indir):
                raise FileNotFoundError(f"Cache directory {indir} not found.")
        elif self.save_input_to is not None:
            indir = self.save_input_to
            if os.path.isdir(indir):
                for f in os.listdir(indir):
                    os.remove(os.path.join(indir, f))

        if self.save_output_to is not None:
            outdir = self.save_output_to
            if os.path.isdir(outdir):
                for f in os.listdir(outdir):
                    os.remove(os.path.join(outdir, f))

        os.makedirs(indir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        self.workdir = workdir
        self.indir = indir
        self.outdir = outdir

    @rank_zero_only
    def init_data(self):
        # make the dataset on which the FID will be evaluated
        data = instantiate_from_config(self.data_cfg)
        data.prepare_data()
        data.setup()
        dset = data.datasets[self.split]
        self.dset_name = dset.__class__.__name__

        if 0 <= self.num_images < len(dset):
            subindices = np.random.choice(
                np.arange(len(dset)), replace=False, size=(self.num_images,)
            )
            dset = Subset(dset, subindices)

        self.n_data = len(dset)
        self.dloader = DataLoader(
            dset,
            batch_size=data.batch_size,
            num_workers=data.num_workers,
            drop_last=False,
        )

    @rank_zero_only
    def log_single_img(self, img, path):
        img = (img + 1.) / 2.
        img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
        img = img.detach().cpu().numpy()
        img = (255 * img).astype(np.uint8)
        Image.fromarray(img).save(path)

    def on_batch_end(self, trainer, pl_module):
        if self.step_frequency is not None:
            if (
                pl_module.global_step % self.step_frequency == 0
                and pl_module.global_step >= self.min_steps
                and pl_module.global_step not in self.executed_steps
            ):
                self.prepare(logdir=trainer.logdir)
                self.eval_metrics(pl_module)
                self.executed_steps.append(pl_module.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_frequency is not None:
            if (
                pl_module.current_epoch % self.epoch_frequency == 0
                and pl_module.current_epoch >= self.min_epochs
                and pl_module.global_step not in self.executed_steps
            ):
                self.prepare(logdir=trainer.logdir)
                self.eval_metrics(pl_module)
                self.executed_steps.append(pl_module.global_step)

    @rank_zero_only
    def eval_metrics(self, pl_module):
        gs = pl_module.global_step
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        # Input data is always the same and is thus created only once
        indir = self.indir

        if self.keep_intermediate:
            outdir = os.path.join(self.outdir, f"gs-{gs:09}")
            os.mkdir(outdir) # should not overwrite anything
        else:
            outdir = self.outdir # overwrite previous data

        keys = [self.input_key, self.output_key]
        roots = {self.input_key: indir, self.output_key: outdir}

        img_count = {k: 0 for k in keys}
        for batch in tqdm(
            self.dloader,
            desc="Creating images for fidelity scores",
            leave=False,
        ):
            with torch.no_grad():
                # NOTE This requires `log_images` to accept the `to_log` kwarg.
                #      The return value should be a dict containing the
                #      input_key and output_key as keys.
                images = pl_module.log_images(
                    batch, to_log=keys, **self.log_images_kwargs
                )

            for k, save_dir in roots.items():
                if k == self.input_key and (
                    self.input_cached or self.load_input_from is not None
                ):
                    continue

                imgs = images[k]
                if self.clamp:
                    imgs = torch.clamp(imgs, -1., 1.)
                for img in imgs:
                    filepath = os.path.join(save_dir, f"{img_count[k]:06}.png")
                    self.log_single_img(img, filepath)
                    img_count[k] += 1

        scores = calculate_metrics(
            input1=outdir,
            input2=indir,
            isc=self.isc,
            fid=self.fid,
            kid=self.kid,
            verbose=False,
            **self.fid_kwargs
        )

        # Write scores to csv file and log them
        csv_path = os.path.join(self.workdir, "fid.csv")
        with open(csv_path, "a") as f:
            w = csv.writer(f)
            if not self.input_cached:
                # Write header lines
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                w.writerow(["timestamp", now])
                w.writerow(["keys", keys])
                w.writerow(["step", "num_samples"] + list(scores.keys()))
            w.writerow([gs, self.n_data] + list(scores.values()))

        for k, v in scores.items():
            pl_module.log(k, v, logger=True, on_epoch=True)
        if is_train:
            pl_module.train()

        self.input_cached = True # always True after first eval_metrics call


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass



if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: cutlit.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python cutlit.py`
    # (in particular `cutlit.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join(opt.logdir, 'logs', nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        config = resolve_based_on(config)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "cutlit.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "cutlit.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "cutlit.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "cutlit.CUDACallback"
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            f"Setting learning rate to {model.learning_rate:.2e} = "
            f"{accumulate_grad_batches} (accumulate_grad_batches) * {ngpu} "
            f"(num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)"
        )

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank==0:
            print(trainer.profiler.summary())

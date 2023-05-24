"""
Main training script.

1. Loads a config file containing all the model's parameters.
2. Sets up training procedures and initializes model, trainer and optimizers.
3. Trains the model.
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path


from typing import Dict

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import neuralmjp
from neuralmjp.utils.helper import (
    create_instance,
    expand_params,
    get_device,
    load_params,
)

# fallback to debugger on error
# sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option("-d", "--debug", "debug", is_flag=True, default=False)
@click.option("-nc", "--no-cuda", "no_cuda", is_flag=True, default=False)
@click.option(
    "-r",
    "--resume-training",
    "resume",
    is_flag=True,
    default=False,
    help="resume training from the last checkpoint",
)
@click.option(
    "-rf",
    "--resume-from",
    "resume_from",
    type=click.Path(exists=True),
    help="path to checkpoint.pth to resume from",
)
@click.option(
    "-ri",
    "--resume-ignore",
    "resume_ignore",
    multiple=True,
    default=[],
    help="only learn prior parameters",
)
@click.option(
    "-rp",
    "--resume-prior",
    "resume_prior",
    is_flag=True,
    default=False,
    help="resume learning only prior parameters",
)


@click.version_option(neuralmjp.__version__)
def main(
    cfg_path: Path,
    log_level: int,
    debug: bool,
    resume: bool,
    resume_from: str,
    resume_ignore: list,
    resume_prior: list,
    no_cuda: bool,
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    params = load_params(cfg_path, _logger)
    gs_params = expand_params(params)
    if resume_from is None and resume:
        resume_from = get_latest_checkpoint(params)
    train(debug, gs_params, params, resume_from, resume_ignore, resume_prior, no_cuda)


def train(debug, gs_params, params, resume, resume_ignore, resume_prior, no_cuda):
    num_workers = params["num_workers"]
    world_size = params.get("world_size", 1)
    distributed = params.get("distributed", False)
    if debug:
        train_in_debug(debug, gs_params, resume, no_cuda)
    elif distributed:
        train_distributed(world_size, gs_params, resume, no_cuda)
    else:
        train_parallel(num_workers, gs_params, resume, resume_ignore, resume_prior, no_cuda)


def train_in_debug(debug, gs_params, resume, no_cuda):
    for search in gs_params:
        train_params(search, resume, debug, no_cuda)


def train_parallel(num_workers, gs_params, resume, resume_ignore, resume_prior, no_cuda):
    p = Pool(num_workers)
    p.map(partial(train_params, resume=resume, resume_ignore=resume_ignore, resume_prior=resume_prior, no_cuda=no_cuda), gs_params)


def train_distributed(world_size, gs_params, resume, no_cuda):
    for param in gs_params:
        mp.spawn(
            train_params_distributed,
            args=(world_size, param, no_cuda),
            nprocs=world_size,
            join=True,
        )


def train_params_distributed(rank, world_size, params, no_cuda):
    if rank == 0:
        _logger.info(f"Name of the Experiment: {params['name']} on rank {rank}")
    print(
        f"Name of the Experiment: {params['name']} on rank {rank} and world size {world_size}"
    )
    setup(rank, world_size)
    device = get_device(params, rank, _logger, no_cuda=no_cuda)
    data_loader = create_instance("data_loader", params, device, rank, world_size)

    model = create_instance("model", params)
    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    trainer = create_instance(
        "trainer",
        params,
        model,
        optimizers,
        True,  # distributed
        False,  # resume
        params,
        data_loader,
    )
    best_model = trainer.train()
    with open(
        os.path.join(params["trainer"]["logging"]["logging_dir"], "best_models.txt"),
        "a+",
    ) as f:
        f.write(str(best_model) + "\n")


def train_params(params, resume, resume_ignore, resume_prior, debug=False, no_cuda: bool = False):
    if debug:
        torch.manual_seed(params["seed"])
    _logger.info("Name of the Experiment: " + params["name"])
    device = get_device(params, no_cuda=no_cuda)
    data_loader = create_instance("data_loader", params, device)

    dl_params = data_loader.get_params()
    params["model"]["args"].update(dl_params)

    model = create_instance("model", params)

    # Optimizers
    optimizers = init_optimizer(model, params, resume_prior)

    # Trainer
    trainer = create_instance(
        "trainer", params, model, optimizers, False, resume, resume_ignore, params, data_loader
    )
    best_model = trainer.train()
    with open(
        os.path.join(params["trainer"]["logging"]["logging_dir"], "best_models.txt"),
        "a+",
    ) as f:
        f.write(str(best_model) + "\n")


def init_optimizer(model, params, resume_prior):
    optimizers = dict()

    # split off model parameters used for generating prior parameters
    if resume_prior:
        non_prior_parameters = None
        if "optimizer_prior" in params:
            all_model_params = None
            prior_parameters = list(model.mastereqencoder.latent_process.get_prior_params.parameters())
        else:
            all_model_params = list(model.mastereqencoder.latent_process.get_prior_params.parameters())
    else:
        all_model_params = list(model.parameters())
        prior_parameters = list(model.mastereqencoder.latent_process.get_prior_params.parameters())
        non_prior_parameters = [p for p in model.parameters() if
                            p not in set(model.mastereqencoder.latent_process.get_prior_params.parameters())]

    # general optimizer
    general_optimizer_params = non_prior_parameters if "optimizer_prior" in params else all_model_params

    if general_optimizer_params is not None:
        optimizer = create_instance("optimizer", params, general_optimizer_params)
        optimizers["optimizer"] = {
            "opt": optimizer,
            "grad_norm": params["optimizer"].get("gradient_norm_clipping", None),
            "min_lr_rate": params["optimizer"].get("min_lr_rate", 1e-8),
        }

    # prior optimizer
    if "optimizer_prior" in params:
        optimizer_prior = create_instance("optimizer_prior", params, prior_parameters)
        optimizers["optimizer_prior"] = {
            "opt": optimizer_prior,
            "grad_norm": params["optimizer_prior"].get("gradient_norm_clipping", None),
            "min_lr_rate": params["optimizer_prior"].get("min_lr_rate", 1e-8),
        }

    return optimizers


def get_latest_checkpoint(params: Dict, best_model: bool = False) -> str:
    save_dir = os.path.join(params["trainer"]["save_dir"], params["name"])
    if not os.path.exists(save_dir):
        raise FileNotFoundError()
    latest_run = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
    if best_model and os.path.exists(os.path.join(latest_run, "best_model.pth")):
        return os.path.join(latest_run, "best_model.pth")
    checkpoints = [x for x in os.listdir(latest_run) if x.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files in directory {latest_run}.")
    latest_checkpoint = sorted(checkpoints)[-1]
    return os.path.join(save_dir, latest_run, latest_checkpoint)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

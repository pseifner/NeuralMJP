from abc import ABC, abstractmethod
from typing import Any
from neuralmjp.utils.helper import create_class_instance, create_instance

import torch
import torch.nn as nn

from ..utils.helper import (
    clip_grad_norm,
    create_instance,
)


class AModel(nn.Module, ABC):
    """
    Base class for model.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def new_stats(self) -> dict:
        """
        Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.
        :return:
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def train_step(
        self, minibatch: list, optimizer: dict, step: int, scheduler: Any = None
    ) -> dict:
        raise NotImplementedError("The train_step method is not implemented in your class!")

    @abstractmethod
    def validate_step(self, minibatch: list) -> dict:
        raise NotImplementedError("The validate_step method is not implemented in your class!")

    @property
    def device(self):
        return next(self.parameters()).device


class VAEMJP(AModel):
    """
    Scaffold for neural variational inference for markov jump processes.
    Initializes mastereq-encoder and decoder, gathers loss terms and sets up monitoring.
    Defines procedures for training, test and validation.
    """

    def __init__(self, **kwargs):
        super(VAEMJP, self).__init__(**kwargs)
        # input; largest observation time (depends on normalization of data)
        self.data_dim = kwargs.get("data_dim")  # data_dim
        self.max_obs_time = kwargs.get("max_obs_time", 1)
        self.time_scaling = kwargs.get("time_scaling", 1)

        # posterior process
        self.n_states = kwargs.get("n_states")  # number of states in latent space
        self.n_proc = kwargs.get("n_proc")  # number of processes in latent space

        # prior process
        self.n_prior_params = kwargs.get("n_prior_params")  # number of prior params

        # global setup for all MLPs
        self.layer_normalization = kwargs.get("layer_normalization", True)
        self.dropout = kwargs.get("dropout", 0.0)

        # Change optimization schedule, for example to train encoder and decoder separately
        self.optimize_prior_separately = kwargs.get("optimize_prior_separately", False)

        # Initial training on small part of time series
        self.pretrain_period_steps = kwargs.get("pretrain_period_steps", 0)  # period length
        self.pretrain_ts_length = kwargs.get("pretrain_ts_length", 10)
        self.pretrain_annealing_steps = kwargs.get("pretrain_annealing_steps", 0)
        self.pretrain_random_indices = kwargs.get("pretrain_random_indices", False)

        # Mastereqencoder and decoder get populated in child classes of this class
        self.mastereqencoder = create_instance(
            "mastereqencoder",
            kwargs,
            self.data_dim,
            self.n_states,
            self.n_proc,
            self.n_prior_params,
            self.layer_normalization,
            self.dropout,
            self.max_obs_time,
            self.time_scaling,
        )
        self.decoder = create_instance(
            "decoder",
            kwargs,
            self.data_dim,
            self.n_states,
            self.n_proc,
            self.layer_normalization,
            self.dropout,
        )

    def forward(self, input, tau):
        """
        Encode data into posterior process. Sample solution at observation times. Gather parameters for decoder.
        input: (x, t, dt), [B, T, data_dim], [B, T, 1], [B, T, 1], observation values, times and delta_times
        tau: [1], sampling temperature for marginal posterior probabilities
        """
        sample_param, q_obs, q0 = self.mastereqencoder(input, tau)

        dec_param = self.decoder(sample_param)

        return dec_param, sample_param, q_obs, q0

    def train_step(
        self, minibatch: dict, optimizer: dict, step: int, scheduler: Any = None
    ) -> dict:
        x = minibatch["observations"].float()
        t = minibatch["obs_times"].float()
        dt = minibatch["delta_times"].float()
        if "mask" in minibatch:
            mask = minibatch["mask"].float()  # [B, T, data_dim]
        else:
            mask = None

        # Pretraining
        seq_len = x.shape[1]
        if seq_len >= self.pretrain_ts_length:
            if step < self.pretrain_period_steps:
                if self.pretrain_random_indices:
                    indices = torch.randperm(seq_len)[: self.pretrain_ts_length]
                    indices, _ = torch.sort(indices)
                else:
                    indices = torch.arange(
                        start=0, end=self.pretrain_ts_length, step=1, device=self.device
                    )
                x = x[:, indices]
                t = t[:, indices]
                dt = dt[:, indices]
                if mask is not None:
                    mask = mask[:, indices]
            elif (
                self.pretrain_period_steps
                <= step
                < self.pretrain_period_steps + self.pretrain_annealing_steps
            ):
                end = int(
                    self.pretrain_ts_length
                    + (step - self.pretrain_period_steps)
                    / self.pretrain_annealing_steps
                    * (x.shape[1] - self.pretrain_ts_length)
                )
                if self.pretrain_random_indices:
                    indices = torch.randperm(seq_len)[:end]
                    indices, _ = torch.sort(indices)
                else:
                    indices = torch.arange(start=0, end=end, step=1, device=self.device)
                x = x[:, indices]
                t = t[:, indices]
                dt = dt[:, indices]
                if mask is not None:
                    mask = mask[:, :end]

        input = (x, t, dt)

        # Schedulers
        if scheduler is not None:
            l_kl = torch.tensor(scheduler["kl_scheduler"](step), device=self.device)
            tau = torch.tensor(scheduler["temperature_scheduler"](step), device=self.device)
        else:
            l_kl = torch.tensor(1.0, device=self.device)
            tau = torch.tensor(1.0, device=self.device)

        schedulers = (l_kl, tau)

        # use optimization schedule determined by config yaml
        loss_stats = self.optimization(optimizer, input, schedulers, mask)

        return loss_stats

    def optimization(self, optimizer, input, schedulers, mask):
        x, t, dt = input
        l_kl, tau = schedulers

        # Setup
        stats = self.new_stats()
        if "optimizer" in optimizer:
            optimizer["optimizer"]["opt"].zero_grad()
        if "optimizer_prior" in optimizer:
            optimizer["optimizer_prior"]["opt"].zero_grad()

        # Build up gradients based on different loss terms
        # Add Reconstruction loss
        self.free_params(self.mastereqencoder)
        self.free_params(self.decoder)
        dec_param, sample_param, q_obs, q0 = self.forward((x, t, dt), tau)

        loss = -self.decoder.get_reconstruction_loss(x, dec_param, q_obs, mask)
        stats["Rec-Loss"] = loss

        # Add KL if necessary
        if l_kl == 0.0:
            loss.backward()
            kl = torch.zeros(1, device=self.device)
        else:
            loss.backward(retain_graph=True)

            if self.optimize_prior_separately:
                self.frozen_params(self.mastereqencoder)
                self.frozen_params(self.decoder)
                self.free_params(self.mastereqencoder.latent_process)
                self.frozen_params(self.mastereqencoder.latent_process._get_rates)
                # don't learn posterior rates on KL

            kl = self.mastereqencoder.get_kl(q0)
            (l_kl * kl).backward()
        stats["KL-Loss"] = kl

        # Apply optimization step
        if "optimizer" in optimizer:
            clip_grad_norm(self.parameters(), optimizer["optimizer"])
            optimizer["optimizer"]["opt"].step()
        if "optimizer_prior" in optimizer and l_kl != 0:
            clip_grad_norm(self.parameters(), optimizer["optimizer_prior"])
            optimizer["optimizer_prior"]["opt"].step()

        # Record parameters
        stats["Combined-Loss"] = loss + l_kl * kl
        stats["l_kl"] = l_kl
        stats["tau"] = tau

        # Add stats from latent process
        stats = self.mastereqencoder.latent_process.add_stats(stats)

        return stats

    def validate_step(self, minibatch: dict) -> dict:
        x = minibatch["observations"].float()
        t = minibatch["obs_times"].float()
        dt = minibatch["delta_times"].float()
        if "mask" in minibatch:
            mask = minibatch["mask"].float()  # [B, T, data_dim]
        else:
            mask = None

        # Schedulers
        l_kl = torch.tensor(1.0, device=self.device)
        tau = torch.tensor(1.0, device=self.device)

        # Compute all losses
        dec_param, sample_param, q_obs, q0 = self.forward((x, t, dt), tau)
        rec_loss = -self.decoder.get_reconstruction_loss(x, dec_param, q_obs, mask)
        kl = self.mastereqencoder.get_kl(q0)

        # Manually record parameters
        stats = self.new_stats()
        stats["Rec-Loss"] = rec_loss
        stats["KL-Loss"] = kl
        stats["Combined-Loss"] = rec_loss + l_kl * kl
        stats["l_kl"] = l_kl
        stats["tau"] = tau

        # Add stats from latent process
        stats = self.mastereqencoder.latent_process.add_stats(stats)

        return stats

    @staticmethod
    def free_params(module):
        if type(module) is not list:
            module = [module]
        for m in module:
            for p in m.parameters():
                p.requires_grad = True

    @staticmethod
    def frozen_params(module):
        if type(module) is not list:
            module = [module]
        for m in module:
            for p in m.parameters():
                p.requires_grad = False

    @staticmethod
    def new_stats() -> dict:
        stats = dict()
        return stats

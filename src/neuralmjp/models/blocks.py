import torch
import numpy as np
from torch import nn as nn
from neuralmjp.models.base_blocks import (
    BlockODE,
    MLP,
    BaseEncoder,
    BaseLatentProcess,
    BaseDecoder,
)
from neuralmjp.utils.helper import gumbel_softmax
from importlib import import_module


class FourierTimeEmbedding(nn.Module):
    """
    From "Self-attention with functional Time Representation Learning", Xu et al. 2019

    n_freqs: number of frequencies
    n_comps: length of truncated fourier series per frequency
    """

    def __init__(self, n_freqs, n_comps):
        super(FourierTimeEmbedding, self).__init__()
        self.twopi = 2 * torch.pi
        self.fs = nn.Parameter(torch.zeros((n_freqs,)))  # frequencies
        self.c1 = nn.Parameter(torch.zeros((n_freqs,)))  # first fourier coefficient (constant)
        self.cs = nn.Parameter(torch.zeros((n_freqs, 2 * n_comps)))  # fourier coefficients

        js = torch.hstack([torch.arange(0, n_comps) + 1, torch.arange(0, n_comps) + 1])  # FFT roots
        ofs = torch.hstack([torch.full((n_comps,), 0), torch.full((n_comps,), self.twopi / 4)])  #

        self.register_buffer("js", js, persistent=False)
        self.register_buffer("ofs", ofs, persistent=False)

    def forward(self, ts):
        """
        ts: [B,*, 1] points in absolute time
        return: [B,* , (2*n_comps + 1) * n_freqs]
        """
        ts = ts.squeeze(-1)
        es = torch.cos(self.twopi * torch.einsum("bk,i,j->bkij", ts, self.fs, self.js) + self.ofs)
        es = torch.einsum("bkij,ij->bkij", es, self.cs)

        c1_b = self.c1.expand((ts.shape[0], ts.shape[1], self.c1.shape[0]))
        out = torch.cat([es.flatten(2, 3), c1_b], dim=-1)

        return out


class MasterEq(BlockODE):
    """
    Solves master equation for the posterior model.
    Computes KL and regularization terms.
    Computes posterior probabilities at observation times and samples these distributions for reconstruction.
    """

    def __init__(
        self,
        data_dim,
        n_states,
        n_proc,
        n_prior_params,
        layer_normalization,
        dropout,
        max_obs_time=1.0,
        time_scaling=1,
        **kwargs
    ):
        super(MasterEq, self).__init__(**kwargs)
        self.data_dim = data_dim  # data-dimension
        self.n_states = n_states  # number of states in latent space
        self.n_proc = n_proc  # number of processes in latent space
        self.layer_normalization = layer_normalization  # layer normalization of MLPs
        self.dropout = dropout
        self.max_obs_time = max_obs_time  # maximum observation time in data

        # use different master equation implementation for regularly sampled observations
        self.irregularly_sampled_obs = kwargs.get("irregularly_sampled_obs", True)

        # use gumbel-softmax to sample posterior marginal distributions
        hard_samples = kwargs.get("hard_samples", True)
        self.gumbel_sample_index = 1 if hard_samples else 0

        # for normalization
        self.q_cutoff = kwargs.get("q_cutoff", 1e-10)
        norm_q_in_me = kwargs.get("norm_q_in_me", False)
        self.norm_q = self.normalize_q if norm_q_in_me else lambda x: x

        # quadrature setup for integral summand of KL
        t_int, weight_quad = self.set_gauss_quadrature(**kwargs)
        self.register_buffer("t_int", t_int, persistent=False)
        self.register_buffer("weight_quad", weight_quad, persistent=False)

        # reoccurring tensors initialized once
        self.register_buffer(
            "array_of_states",
            torch.arange(start=0, end=n_states, step=1, dtype=torch.float),
            persistent=False,
        )

        # Instance of (child of) class BaseEncoder; encodes time and data into some hidden space
        encoder_kwargs = kwargs["encoder"]
        encoder_module = import_module(encoder_kwargs["module"])
        encoder_class = getattr(encoder_module, encoder_kwargs["name"])
        self.encoder = encoder_class(
            data_dim,
            n_states,
            n_proc,
            self.layer_normalization,
            dropout,
            max_obs_time,
            **encoder_kwargs["args"]
        )

        # Expose its time-embedding method and target dimension
        time_data_rep, time_data_rep_dim = (
            self.encoder.get_time_data_rep(),
            self.encoder.get_time_data_rep_dim(),
        )

        # Instance of (child of) class BaseLatentProcess; gets rates of prior process and mean-field approximation
        latent_process_kwargs = kwargs["latent_process"]
        latent_process_module = import_module(latent_process_kwargs["module"])
        latent_process_class = getattr(latent_process_module, latent_process_kwargs["name"])
        self.latent_process = latent_process_class(
            n_states,
            n_proc,
            n_prior_params,
            time_data_rep,
            time_data_rep_dim,
            self.layer_normalization,
            dropout,
            time_scaling,
            **latent_process_kwargs["args"]
        )

        # gather parameters that influence the master eq; odeint_adjoint needs these to assemble adjoint-system
        self.me_adjoint_params = (
            self.encoder.get_me_adjoint_params() + self.latent_process.get_me_adjoint_params()
        )

    def forward(self, input, tau):
        """
        Encodes data observation into a Markov Jump Process
        input: tuple(x: [B, T, data_dim + 2],       --- observation values
                     t: [B, T, 1],                  --- observation times
                     dt: [B, T, 1])                 --- time between observations
        tau: [1]                                    --- temperature for sampling
        return: tuple(
            sample_param: (
                    [B, T, n_proc, 1],              --- sample of reconstruction
                    [B, T, n_proc, n_states ],      --- sample of reconstruction as one-hot vectors
                    )
            q_obs: [B, T, n_proc, n_states]         --- marginal posterior probabilities at observation times
            q0:   [B, n_proc, n_states]             --- marginal distribution at t=0
                    )
        """

        x, t, dt = input
        x = torch.cat([x, t, dt], dim=-1)

        # encode data into hidden representation and get initial distribution
        q0 = self.encoder(x)  # [B, n_proc, n_states]

        # solve posterior master equation
        q_obs = self.get_qs_at_obs(t, q0)

        # sample posterior process at observation times
        sample_param = self.sample_process(q_obs, tau)
        # (z, z_one_hot), ([B, T, n_proc, 1], [B,T,n_proc,n_states])

        return sample_param, q_obs, q0

    def set_gauss_quadrature(self, **kwargs):
        """
        Return weights and abscissae of gauss quadrature for integrals over [0,1]
        return: tuple(
                [T]     --- abscissae of gauss quadrature
                [T]     --- weights of gauss quadrature
        )
        """
        n_steps = kwargs.get("n_steps_quadrature", 50)
        x, w = np.polynomial.legendre.leggauss(n_steps)

        t_int = torch.from_numpy(0.5 * self.max_obs_time * (x + 1.0)).float()
        weight_quad = torch.from_numpy(w)

        return t_int, weight_quad

    def get_qs_at_obs(self, t, q0):
        """
        Solves master equation and returns solution at observation times.
        t: [B, T, 1]                        --- observation times
        q0: [B, n_proc, n_states]           --- initial conditions
        return: [B, T, n_proc, n_states]    --- posterior distribution at observation points
        """
        batch_size, seq_len, _ = t.shape
        t = torch.cat(
            [torch.zeros((batch_size, 1, 1), device=self.device), t], dim=-2
        )  # [B, T+1, 1]

        q0 = q0.unsqueeze(1)

        # solve (possibly SCALED) Master eq. at observation points
        if self.irregularly_sampled_obs:
            q = q0
            q_obs = [q0]
            t_ = torch.tensor([0.0, 1], device=self.device)
            for i in range(seq_len):
                t_0, t_1 = t[:, i], t[:, i + 1]
                state = (t_0, t_1, q)
                _, _, sol = self._call_odeint(
                    self.scaled_master_eq, state, t_, adjoint_params=self.me_adjoint_params
                )
                q = sol[-1]
                q_obs.append(q)
            q_obs = torch.stack(q_obs, dim=0)

        else:
            t = t[0, :, 0]
            if t[1] <= 0.0:
                t[0] = -0.001
                # if regularly sampled often 0 first obs point; causes problems with solver
            q_obs = self._call_odeint(self.master_eq, q0, t, adjoint_params=self.me_adjoint_params)

        q_obs = q_obs[1:].squeeze(2)  # drop t=0 solution and remove unsqueezed dimension
        q_obs = q_obs.permute(1, 0, 2, 3)

        # normalize qs for quadrature, correcting numerical errors, so they can be sampled from
        q_obs = self.normalize_q(q_obs)

        return q_obs

    def master_eq(self, t, q):
        """
        Returns the right-hand side of the master equation dqdt.
        q: [B, n_proc, n_states]
        t: [1]
        returns: [B, n_proc, n_states]
        """
        batch_size = q.shape[0]
        t = t.view(1, 1, 1).expand(batch_size, 1, 1)  # [B, 1, 1]

        q = self.norm_q(q)
        dqdt = self.latent_process.get_dqdt(q, t)

        return dqdt

    def scaled_master_eq(self, t, state):
        """
        Returns the right-hand side of the SCALED master equation dqdt.
        Idea from "Neural Spatiio-Temporal Point Processes", Appendix F.
        Here, t \in (0,1).
        Integral in (t_0, t_1) gets rescaled to (0,1)
        state = (t_0, t_1, q)
        t_0, t_1: [B, 1]
        q: [B, 1, n_proc, n_states]
        t: [1]
        returns: [B, n_proc, n_states]
        """
        t_0, t_1, q = state
        t_0, t_1 = t_0.unsqueeze(1), t_1.unsqueeze(1)
        t = t * (t_1 - t_0) + t_0  # rescale time

        q = self.norm_q(q)
        dqdt = self.latent_process.get_dqdt(q, t)
        dqdt = dqdt * (t_1 - t_0).view(-1, 1, 1, 1)  # rescales back

        return torch.zeros_like(t_0), torch.zeros_like(t_1), dqdt

    def get_kl(self, q0, return_mean=True):
        """
        Compute KL loss using gaussian quadrature.
        q0:    [B, n_proc, n_states]
        return_mean: Bool
        returns: [1]    --- KL for batch or each time series
        """
        # solve Master eq. at integration points
        q0 = q0.unsqueeze(1)
        # this is neede for latent_process.get_dqdt to work; [B, 1, n_proc, n_states]

        t_ = torch.cat([torch.zeros(1, device=self.device), self.t_int])  # [T+1]

        # get posterior marginal distribution at quadrature times
        q_int = self._call_odeint(self.master_eq, q0, t_, adjoint_params=self.me_adjoint_params)

        q_int = q_int[1:].squeeze(2)  # drop t=0 solution and remove unsqueezed dimension
        q_int = q_int.permute(1, 0, 2, 3)  # [B, n_steps_quadrature, n_proc, n_states]

        q_int = self.normalize_q(q_int)

        # set up quadrature
        batch_size = q0.shape[0]

        t_int = self.t_int.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [B, t_int, 1]
        w_int = self.weight_quad.view(1, -1)

        # get kl from latent process
        kl_integrand = self.latent_process.get_kl_integrand(t_int, q_int)

        # apply quadrature rule to kl
        kl = (self.max_obs_time / 2.0) * torch.sum(w_int * kl_integrand, dim=-1)

        if return_mean:
            return kl.mean()
        else:
            return kl

    def sample_process(self, q, tau):
        """
        Sample from the aggregated distribution at each observation time
        q: [B, T, n_proc, n_states]
        Returns samples with shape [B, T, n_proc, n_states]
            and one_hot tensors they were sampled with
        """
        z_one_hot = gumbel_softmax(q, tau, self.device)  # tuple of soft and hard sample
        z_one_hot = z_one_hot[self.gumbel_sample_index]

        z = torch.sum(z_one_hot * self.array_of_states.view(1, 1, 1, -1), dim=-1)
        z = z.unsqueeze(-1)

        return z, z_one_hot

    def normalize_q(self, q):
        """
        Returns proper probability distributions.
        Standard normalization: clamp qs to small nonzero value and normalize tensor
        """
        q = q.clamp(min=self.q_cutoff)
        q = q / (q.sum(dim=-1, keepdim=True))
        return q


class ODERNN(BaseEncoder):
    """
    Encodes observations into a single hidden state with ODERNN (GRU updates at every observation).
    Learns initial distribution for posterior process based on the single hidden state.
    Provides time-data representation that simply concatenates a Fourier Time Embedding to the single hidden state.
    """

    def __init__(
        self, data_dim, n_states, n_proc, layer_normalization, dropout, max_obs_time, **kwargs
    ):
        super(ODERNN, self).__init__(
            data_dim, n_states, n_proc, layer_normalization, dropout, max_obs_time, **kwargs
        )

        # encode data into hidden space with latent ODE
        self.hidden_dim = kwargs.get("hidden_dim", 128)

        # options for neural ODE
        layers_ode = kwargs.get("layers_ode")
        activation_ode = kwargs.get("activation_ode")
        init_ode = kwargs.get("init_ode")
        self.ode = MLP(
            self.hidden_dim,
            layers_ode,
            self.hidden_dim,
            activation=activation_ode,
            layer_normalization=self.layer_normalization,
            init_method=init_ode,
            dropout=self.dropout,
        )
        self.gru_cell = nn.GRUCell(
            input_size=data_dim + 2, hidden_size=self.hidden_dim, dtype=torch.float
        )

        self.init_solver_time = kwargs.get("init_solver_time", 0.1)

        # Steer regularization from https://arxiv.org/abs/2006.10711
        self.use_steer = kwargs.get("use_steer", True)
        self.steer_eps = kwargs.get("steer_eps", 1e-6)

        # save hidden representations after each forward
        self.hidden_rep = None

        # options for mlp for initial posterior distribution:
        layers_mlp_q0 = kwargs.get("layers_mlp_q0")
        activation_q0 = kwargs.get("activation_q0")
        init_mlp_q0 = kwargs.get("init_mlp_q0", None)
        self._get_logits = MLP(
            self.hidden_dim,
            layers_mlp_q0,
            n_states * n_proc,
            activation=activation_q0,
            layer_normalization=self.layer_normalization,
            init_method=init_mlp_q0,
            dropout=self.dropout,
        )

        # encode time itself with fourier time embeddings (mercer time embeddings)
        n_freqs = kwargs.get("n_freqs", 5)
        n_comps = kwargs.get("n_comps", 5)
        fte_dim = (2 * n_comps + 1) * n_freqs

        self.use_fourier_time_embedding = kwargs.get("use_fourier_time_embedding", False)
        self.time_map = (
            FourierTimeEmbedding(n_freqs=n_freqs, n_comps=n_comps)
            if self.use_fourier_time_embedding is True
            else (lambda x: x)
        )

        # Dimension of and parameters used to get time-data embedding h(t)
        self.time_data_rep_dim = (
            fte_dim + self.hidden_dim
            if self.use_fourier_time_embedding is True
            else 1 + self.hidden_dim
        )
        self.me_adjoint_params = tuple(self.ode.parameters()) + tuple(self.gru_cell.parameters())
        if self.use_fourier_time_embedding:
            self.me_adjoint_params = self.me_adjoint_params + tuple(self.time_map.parameters())

    def forward(self, x):
        """
        Returns initial distribution and sets up time_data_rep.
        x: [B, T, data_dim + 2]          --- observations, time and time-differences
        return: [B, n_proc, n_states]    --- initial posterior distribution
        """
        # encode data
        self.data_rep(x)

        # initial distribution based on data encoding
        q0 = self.get_initial_distribution()

        return q0

    def data_rep(self, x):
        """
        Encodes data into hidden representation of the series
        x: [B, T, data_dim + 2]     --- concatenated observation values, times and delta-times
        returns: *                  --- data rep that gets passed to get_initial_distribution
        """
        batch_size, seq_length, _ = x.shape
        t = x[:, :, -2]

        # solve ode from self.max_obs_time+self.init_solver_time to last observation and update with gru
        t1 = t[:, -1]
        t0 = torch.ones_like(t1) * (self.max_obs_time + self.init_solver_time)
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        h = self._latent_ode_step(t0, t1, h, gru=True, obs=x[:, -1])

        # solve ode backwards in time and update with gru at every observation time
        for i in reversed(range(1, seq_length)):
            t0, t1 = t[:, i], t[:, i - 1]
            h = self._latent_ode_step(t0, t1, h, gru=True, obs=x[:, i])

        # solve ode to time 0 without updating (no observation)
        t0 = t[:, 0]
        t1 = torch.zeros_like(t0)
        h = self._latent_ode_step(t0, t1, h).unsqueeze(1)

        # last hidden representation is used in time_data_rep and for initial distribution
        self.hidden_rep = h

    def get_initial_distribution(self):
        """
        Returns the initial posterior distribution.
        returns: [B, n_proc, n_states]   --- posterior distribution at t=0
        """
        h = self.hidden_rep
        batch_size = h.size(0)

        logits = self._get_logits(h)  # [B, n_states * n_proc]
        logits = logits.view(batch_size, self.n_proc, -1)

        return torch.nn.functional.softmax(logits, dim=-1)

    def time_data_rep(self, t):
        """
        Continuous time embedding of observations at time t based on value of ODERNN at time t=0.
        t: [B, *, 1]
        return: [B, *, time_data_rep_dim]   --- time-data embedding at t
        """
        seq_length = t.shape[1]

        # time embedding by time_map
        t = self.time_map(
            t
        )  # [B, *, fte_dim] or [B, *, 1], depending on self.use_fourier_time_embedding

        # data embedding is h(0) or the final value of backwards Latent ODE
        h = self.hidden_rep.expand(-1, seq_length, -1)  # [B, *, hidden_dim]

        batch_size = h.shape[0]
        t = t.expand(batch_size, -1, -1)

        # concatenate time and data embedding
        embd = torch.cat([t, h], dim=-1)  # [B, * , time_data_rep_dim]

        return embd

    def _latent_ode_step(self, t0, t1, h, gru=False, obs=None):
        """
        Solve ODERNN in [t0,t1] with initial condition h, take solution at [t1], possibly update with gru and return its output.
        t0, t1: [B]
        h: [B, hidden_dim]
        gru: Bool
        obs: [B, D]
        return: [B, hidden_dim]
        """
        if self.use_steer is True:
            b = torch.abs(t1 - t0) - self.steer_eps
            r = torch.randn_like(b) * 2 - 1.0  # ~ U([-1,1])
            t1 = t1 + r * b

        t_ = torch.tensor([1.0, 0], device=self.device)
        state = (t0, t1, h)
        _, _, sol = self._call_odeint(
            self._ode_forward, state, t_, adjoint_params=tuple(self.ode.parameters())
        )

        h = sol[-1].float()  # [B, hidden_dim]
        if gru is True:
            h = self.gru_cell(obs, h)

        return h

    def _ode_forward(self, s, state):
        """
        (Scaled) forward of the neural ODE.
        """
        t0, t1, x = state
        ratio = (t1 - t0) / (1 - 0)
        out = self.ode(x)
        out = out * ratio.unsqueeze(1)

        return torch.zeros_like(t0), torch.zeros_like(t1), out


class LV(BaseLatentProcess):
    """
    Prior and posterior rates for a Lotka-Volterra model and the associated KL loss.
    Mean-field approach for posterior process and also restricts posterior process to a pure birth-death process
    """

    def __init__(
        self,
        n_states,
        n_proc,
        n_prior_params,
        time_data_rep,
        time_data_rep_dim,
        layer_normalization,
        dropout,
        time_scaling,
        **kwargs
    ):
        super(LV, self).__init__(
            n_states,
            n_proc,
            n_prior_params,
            time_data_rep,
            time_data_rep_dim,
            layer_normalization,
            dropout,
            time_scaling,
            **kwargs
        )
        # options for mlp for mean-field rates
        layers_mlp_rates = kwargs.get("layers_mlp_rates")
        activation_mlp_rates = kwargs.get("activation_mlp_rates")
        init_mlp_rates = kwargs.get("init_mlp_rates", None)

        self._get_rates = MLP(
            time_data_rep_dim,
            layers_mlp_rates,
            2 * (n_states - 1) * 2,
            activation=activation_mlp_rates,
            layer_normalization=self.layer_normalization,
            init_method=init_mlp_rates,
            dropout=self.dropout,
        )

        # apply clamp(min=self.rate_cutoff) before applying log to any rates (to avoid log(0))
        self.rate_cutoff = kwargs.get("rate_cutoff", 1e-6)

        # for kl_integrand
        self.register_buffer(
            "state_vector", torch.arange(self.n_states).float().view(1, 1, -1), persistent=False
        )  # [1, 1, n_states]
        self.register_buffer(
            "state_matrix",
            torch.outer(torch.arange(self.n_states), torch.arange(self.n_states))
            .float()
            .view(1, 1, self.n_states, self.n_states),
            persistent=False,
        )

        self.me_adjoint_params = tuple(self._get_rates.parameters())

    def get_rates(self, t: torch.Tensor):
        """
        Returns mean-field transition rates at t.
        t: [B, *, 1]    --- time
        output: [B, *, (n_states - 1) * 2 * 2]
        """
        seq_length = t.shape[1]

        embd = self.time_data_rep(t)  # h(t)
        g = self._get_rates(embd)  # [B, *, (n_states - 1) * 2 * 2]
        g = g.view(-1, seq_length, 2, 2, self.n_states - 1)

        return torch.exp(g)

    def get_dqdt(self, q, t):
        """
        Returns the derivative of the (local/aggregated) distribution over states
        for each process
        q: [B, 1, 2, n_states]
        t: [B, 1, 1]
        return: [B, 1, 2, n_states] --- dq/dt
        """
        g = self.get_rates(t)

        g_u = g[:, :, 0]
        g_d = g[:, :, 1]

        zeros = torch.zeros_like(g_u)[..., 0].unsqueeze(-1)

        # gq_u[..., 0] is rate of jumping up (multiplied by its state probability) out of 0-th state (population)
        gq_u = torch.cat([g_u, zeros], dim=-1) * q
        gq_d = torch.cat([zeros, g_d], dim=-1) * q

        return (
            torch.cat([zeros, gq_u[..., :-1]], dim=-1)
            + torch.cat([gq_d[..., 1:], zeros], dim=-1)
            - (gq_u + gq_d)
        )  #  [B, T, 2, n_states]

    def get_kl_integrand(self, t, q):
        """
        Computes integrand of Kullback Leibler distance of Lotka-Volterra process and the posterior process (birth-death, mean-field).
        This method assumes that both processes are represented by transition rates.
        t: [B, T, 1]                                            --- quadrature times
        q: [B, T, 2, n_states]                                  --- (mean-field) aggregated probability over states
        """
        batch_size, seq_len, _, _ = q.shape

        g = self.get_rates(t)

        # KL integrand
        g_u = g[:, :, 0]  # [B, T, 2, n_states - 1]
        g_d = g[:, :, 1]  # [B, T, 2, n_states - 1]

        g_u, g_d = g_u.clamp(min=self.rate_cutoff), g_d.clamp(min=self.rate_cutoff)

        alpha, beta, gamma, delta = self.prior_params(
            num_samples=batch_size, rescale=False, mean=False
        )

        f_0_u = (alpha.view(-1, 1, 1) * self.state_vector[:, :, :-1]).clamp(min=self.rate_cutoff)
        f_0_d = (beta.view(-1, 1, 1, 1) * self.state_matrix[:, :, 1:, :]).clamp(
            min=self.rate_cutoff
        )
        f_1_u = (delta.view(-1, 1, 1, 1) * self.state_matrix[:, :, :, :-1]).clamp(
            min=self.rate_cutoff
        )
        f_1_d = (gamma.view(-1, 1, 1) * self.state_vector[:, :, 1:]).clamp(min=self.rate_cutoff)

        q_01 = torch.einsum("bti,btj->btij", (q[:, :, 0], q[:, :, 1]))  # [B, T, n_states, n_states]

        kl_integrand = (
            torch.sum(
                q[:, :, :, :-1] * g_u * (torch.log(g_u) - 1.0)
                + q[:, :, :, 1:] * g_d * (torch.log(g_d) - 1.0),
                dim=(-1, -2),
            )
            - torch.sum(
                q[:, :, 0, :-1] * g_u[:, :, 0] * torch.log(f_0_u)
                + q[:, :, 1, 1:] * g_d[:, :, 1] * torch.log(f_1_d),
                dim=-1,
            )
            - torch.sum(
                q_01[:, :, 1:, :] * g_d[:, :, 0].unsqueeze(-1) * torch.log(f_0_d), dim=(-1, -2)
            )
            - torch.sum(
                q_01[:, :, :, :-1] * g_u[:, :, 1].unsqueeze(-2) * torch.log(f_1_u), dim=(-1, -2)
            )
            + torch.sum(q[:, :, 0, :-1] * f_0_u + q[:, :, 1, 1:] * f_1_d, dim=-1)
            + torch.sum(q_01[:, :, 1:, :] * f_0_d, dim=(-1, -2))
            + torch.sum(q_01[:, :, :, :-1] * f_1_u, dim=(-1, -2))
        )  # [B, T]

        return kl_integrand

    def prior_params(self, num_samples=1, rescale=True, mean=False, return_std=False):
        """
        Returns prior parameters.
        num_samples: if self.use_generative_prior_params, sample num_samples prior params
        rescale: if True, return prior parameters rescaled to original time scale
        mean: returns mean parameters over num_samples
        """
        if self.use_generative_prior_params:
            eta = self.input_std_prior_params_mlp * torch.randn(
                (num_samples, self.input_dim_prior_params_mlp), device=self.device
            )
            param = torch.exp(self.get_prior_params(eta))
            std = param.std(dim=0)
            if mean:
                param = param.mean(dim=0)
            alpha, beta, gamma, delta = param[..., 0], param[..., 1], param[..., 2], param[..., 3]
            alpha_std, beta_std, gamma_std, delta_std = std[0], std[1], std[2], std[3]
        else:
            alpha, beta, gamma, delta = torch.exp(self.get_prior_params)

        if rescale is True:
            alpha, beta, gamma, delta = (
                alpha / self.time_scaling,
                beta / self.time_scaling,
                gamma / self.time_scaling,
                delta / self.time_scaling,
            )
            alpha_std, beta_std, gamma_std, delta_std = (
                alpha_std / self.time_scaling,
                beta_std / self.time_scaling,
                gamma_std / self.time_scaling,
                delta_std / self.time_scaling,
            )

        if return_std:
            return alpha, beta, gamma, delta, alpha_std, beta_std, gamma_std, delta_std
        else:
            return alpha, beta, gamma, delta

    def add_stats(self, stats):
        """
        Add entries to loss stats dictionary that are specific to LV.
        """
        alpha, beta, gamma, delta = self.prior_params(num_samples=1000, rescale=True, mean=True)

        stats["alpha"] = alpha
        stats["beta"] = beta
        stats["delta"] = delta
        stats["gamma"] = gamma

        return stats


class DFR(BaseLatentProcess):
    """
    Prior and posterior rates for a discrete flashing ratchet model and the associated KL loss.
    Parametrization of this process is taken from: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.150607
    """

    def __init__(
        self,
        n_states,
        n_proc,
        n_prior_params,
        time_data_rep,
        time_data_rep_dim,
        layer_normalization,
        dropout,
        time_scaling,
        **kwargs
    ):
        super(DFR, self).__init__(
            n_states,
            n_proc,
            n_prior_params,
            time_data_rep,
            time_data_rep_dim,
            layer_normalization,
            dropout,
            time_scaling,
            **kwargs
        )
        # options for mlp for mean-field rates
        layers_mlp_rates = kwargs.get("layers_mlp_rates")
        activation_mlp_rates = kwargs.get("activation_mlp_rates")
        init_mlp_rates = kwargs.get("init_mlp_rates", None)

        self._get_rates = MLP(
            time_data_rep_dim,
            layers_mlp_rates,
            6 * 6,
            activation=activation_mlp_rates,
            layer_normalization=self.layer_normalization,
            init_method=init_mlp_rates,
            dropout=self.dropout,
        )

        # create masks for prior (and posterior) rates
        # can mask out posterior rates we know to be 0 from prior process
        self.mask_posterior_rates = kwargs.get("mask_posterior_rates", False)
        self.register_buffer(
            "prior_mask",
            torch.tensor(
                [
                    [0.0, 1, 1, 1, 0, 0],
                    [1.0, 0, 1, 0, 1, 0],
                    [1.0, 1, 0, 0, 0, 1],
                    [1.0, 0, 0, 0, 1, 1],
                    [0.0, 1, 0, 1, 0, 1],
                    [0.0, 0, 1, 1, 1, 0],
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "diagonal_mask", (1 - torch.eye(6, 6)).view(1, 1, 1, 6, 6), persistent=False
        )

        # apply clamp(min=self.rate_cutoff) before applying log to any rates (to avoid log(0))
        self.rate_cutoff = kwargs.get("rate_cutoff", 1e-6)

        self.me_adjoint_params = tuple(self._get_rates.parameters())

    def get_rates(self, t: torch.Tensor):
        """
        Returns non-negative rates at t, can either be used as transition rates or marginal density rates
        of posterior process
        t: [B, *, 1]    --- time
        output: [B, *, 6, 6]
        """
        seq_length = t.shape[1]

        # get square rates matrix (g[...,i,j] is rate for transitions from state i to state j)
        embd = self.time_data_rep(t)
        g = self._get_rates(embd)  # [B, *, 6 * 6]
        g = g.view(-1, seq_length, 1, 6, 6)
        g = torch.exp(g)

        # remove rates on diagonal
        g = g * self.diagonal_mask

        if self.mask_posterior_rates is True:
            g = g * self.prior_mask

        return g

    def prior_params(self, num_samples=1, rescale=True, mean=False, return_std=False):
        """
        Returns prior parameters.
        num_samples: if self.use_generative_prior_params, sample num_samples prior params
        rescale: if True, return prior parameters rescaled to original time scale
        mean: returns mean parameters over num_samples
        """

        if self.use_generative_prior_params:
            eta = self.input_std_prior_params_mlp * torch.randn(
                (num_samples, self.input_dim_prior_params_mlp), device=self.device
            )
            param = torch.exp(self.get_prior_params(eta))
            std = param.std(dim=0)
            if mean:
                param = param.mean(dim=0)
            V, r, b = param[..., 0], param[..., 1], param[..., 2]
            V_std, r_std, b_std = std[0], std[1], std[2]
        else:
            V, r, b = torch.exp(self.get_prior_params)

        if return_std:
            return V, r, b, V_std, r_std, b_std
        else:
            return V, r, b

    def get_prior_trans(self, batch_size):
        """
        Build transition rate matrix K of a discrete flashing ratchet model,
        where k_ij = trans rate from i to j (in all_states tensor).
        Will multiply K from right onto identified_states
        """
        V, r, b = self.prior_params(num_samples=batch_size, rescale=False, mean=False)

        V = torch.tensor([0, 1, 2], device=self.device) * V.view(-1, 1)
        V_j, V_i = V.view(batch_size, 1, 3), V.view(batch_size, 3, 1)

        # rates for changes in x when eta == 0
        mask = torch.ones((batch_size, 3, 3), device=self.device) - torch.eye(
            3, device=self.device
        )  # [B,3,3]
        rates_x_eta_0 = torch.exp(-1 / 2 * (V_j - V_i)) * mask

        # rates for changes in x when eta == 1
        rates_x_eta_1 = b.view(-1, 1, 1) * mask

        # rates for changes in eta
        rates_eta = r.view(-1, 1, 1) * torch.eye(3, device=self.device).unsqueeze(0)

        K = torch.cat(
            [
                torch.cat([rates_x_eta_0, rates_eta], dim=-1),
                torch.cat([rates_eta, rates_x_eta_1], dim=-1),
            ],
            dim=1,
        ).unsqueeze(
            1
        )  # [B, 1, 6, 6]

        # Cancel out effect of time normalization on transition rates; needed to compare learned parameters to ground truth parameters
        K = K * self.time_scaling

        return K

    def get_dqdt(self, q, t):
        """
        Returns the derivative of the (local/aggregated) distribution over states.
        q: [B, 1, n_proc, n_states]
        t: [B, 1, 1]
        """
        g = self.get_rates(t)

        # fill diagonal with proper rates
        g = g - torch.diag_embed(g.sum(dim=-1))

        return torch.matmul(q.unsqueeze(-2), g).squeeze(-2)  #  [B, T, 1, 6]

    def get_kl_integrand(self, t, q):
        """
        Computes integrand of Kullback Leibler between prior and posterior process of this model, if both are
        represented by transition rates.
        t: [B, T, 1]                                                    --- quadrature times
        q: [B, T, 1, 6]                                                 --- (mean-field) aggregated probability over states
        """
        batch_size, seq_len, _, _ = q.shape

        # get posterior, prior rates
        g = self.get_rates(t)  # [B, T, n_proc, n_states, n_states]
        K = self.get_prior_trans(batch_size)  # [B, 6, 6], [1]
        K = K.view(batch_size, 1, self.n_proc, self.n_states, self.n_states)  # [B, 1, 1, 6, 6]

        # avoid taking log of very small rates
        log_K = torch.log(K.clamp(self.rate_cutoff)) * self.diagonal_mask
        log_g = torch.log(g.clamp(self.rate_cutoff)) * self.diagonal_mask

        sum_ = (g * (log_g - 1 - log_K) + K).sum(dim=-1)  # [B, T, 1, 6]

        kl_integrand = (q * sum_).sum(dim=-1).squeeze(-1)  # [B, T]

        return kl_integrand

    def add_stats(self, stats):
        """
        Add entries to loss stats dictionary that are specific to DFR.
        """
        V, r, b = self.prior_params(num_samples=1000, rescale=False, mean=True)

        stats["V"] = V
        stats["r"] = r
        stats["b"] = b

        return stats


class GenericMeanField(BaseLatentProcess):
    """
    Prior and posterior process consist of n_proc processes with n_states states and have no other restrictions than
    mean-field transition rates applied to them.
    """

    def __init__(
        self,
        n_states,
        n_proc,
        n_prior_params,
        time_data_rep,
        time_data_rep_dim,
        layer_normalization,
        dropout,
        time_scaling,
        **kwargs
    ):
        super(GenericMeanField, self).__init__(
            n_states,
            n_proc,
            n_prior_params,
            time_data_rep,
            time_data_rep_dim,
            layer_normalization,
            dropout,
            time_scaling,
            **kwargs
        )
        layers_mlp_rates = kwargs.get("layers_mlp_rates")
        activation_mlp_rates = kwargs.get("activation_mlp_rates")
        init_mlp_rates = kwargs.get("init_mlp_rates", None)

        self._get_rates = MLP(
            time_data_rep_dim,
            layers_mlp_rates,
            n_proc * n_states * n_states,
            activation=activation_mlp_rates,
            layer_normalization=self.layer_normalization,
            init_method=init_mlp_rates,
            dropout=self.dropout,
        )

        self.register_buffer(
            "diagonal_mask",
            (1 - torch.eye(self.n_states, self.n_states)).view(
                1, 1, 1, self.n_states, self.n_states
            ),
            persistent=False,
        )

        self.rate_cutoff = kwargs.get("rate_cutoff", 1e-6)
        # for kl_integrand
        self.register_buffer(
            "state_vector", torch.arange(self.n_states).float().view(1, 1, -1), persistent=False
        )  # [1, 1, n_states]
        self.register_buffer(
            "state_matrix",
            torch.outer(torch.arange(self.n_states), torch.arange(self.n_states))
            .float()
            .view(1, 1, self.n_states, self.n_states),
            persistent=False,
        )

        self.me_adjoint_params = tuple(self._get_rates.parameters())

    def get_rates(self, t: torch.Tensor):
        """
        Returns non-negative transition rates at t.
        t: [B, *, 1]    --- time
        output: [B, *, n_proc, n_states, n_states]
        """
        B, T, _ = t.shape

        # get square rates matrix (g[...,k,i,j] is rate for transitions from state i to state j in process k)
        embd = self.time_data_rep(t)
        g = self._get_rates(embd)  # [B, *, n_proc * n_states * n_states]
        g = g.view(B, T, self.n_proc, self.n_states, self.n_states)
        g = torch.exp(g)

        # remove rates on diagonal
        g = g * self.diagonal_mask

        return g

    def get_dqdt(self, q, t):
        """
        Returns the derivative of the (local/aggregated) distribution over states.
        q: [B, T, n_proc, n_states]
        t: [B, T, 1]
        """

        g = self.get_rates(t)

        # fill diagonal with rates
        g = g - torch.diag_embed(g.sum(dim=-1))

        return torch.matmul(q.unsqueeze(-2), g).squeeze(-2)  # [B, T, n_proc, n_states]

    def get_kl_integrand(self, t, q):
        """
        Computes integrand of Kullback Leibler between two generic, mean field Markov Jump Processes.
        t: [B, T, 1]                                    --- quadrature times
        q: [B, T, n_proc, n_states]                     --- (mean-field) aggregated probability over states
        """
        batch_size, seq_len, _, _ = q.shape

        # n-fold outer product of marginal probabilities
        qs = torch.unbind(q, dim=2)

        view_list = [
            (batch_size, seq_len) + i * (1,) + (-1,) + (self.n_proc - 1 - i) * (1,)
            for i in range(self.n_proc)
        ]
        final_shape = (batch_size, seq_len) + self.n_proc * (self.n_states,)
        qs = [qs[i].view(view_list[i]).expand(final_shape) for i in range(self.n_proc)]

        qs_stacked = torch.stack(qs, dim=0)
        q = torch.prod(qs_stacked, dim=0)  # [B, T, n_states, n_states, ... , n_states]

        # get posterior and prior rates
        g = self.get_rates(t)  # [B, T, n_proc, n_states, n_states]

        f = self.prior_params(num_samples=batch_size, rescale=False, mean=False)

        # avoid taking log of very small rates
        log_f = torch.log(f.clamp(self.rate_cutoff)) * self.diagonal_mask
        log_g = torch.log(g.clamp(self.rate_cutoff)) * self.diagonal_mask

        summand = g * (log_g - 1 - log_f) + f  # [B, T, n_proc, n_states, n_states]

        # sum rates of transitioning out of a posteriori state of one process (diagonal is 0, so we can sum over dim=-2)
        summand = summand.sum(dim=-1)

        # posterior process can transition only in exactly one dimension, so take outer sum of summand over the n_proc dimension
        summands = torch.unbind(summand, dim=2)
        summands = [summands[i].view(view_list[i]).expand(final_shape) for i in range(self.n_proc)]

        summands_stacked = torch.stack(summands, dim=0)
        summand = torch.sum(summands_stacked, dim=0)  # [B, T, n_states, n_states, ..., n_states]

        # finish calculation of KL integrand
        kl_integrand = q * summand
        kl_integrand = kl_integrand.flatten(start_dim=2).sum(dim=-1)  # [B, T]

        return kl_integrand

    def prior_params(self, num_samples=1, rescale=True, mean=False, return_std=False):
        """
        Returns prior parameters.
        num_samples: if self.use_generative_prior_params, sample num_samples prior params
        rescale: if True, return prior parameters rescaled to original time scale
        mean: returns mean parameters over num_samples
        """

        if self.use_generative_prior_params:
            eta = self.input_std_prior_params_mlp * torch.randn(
                (num_samples, self.input_dim_prior_params_mlp), device=self.device
            )
            f = torch.exp(self.get_prior_params(eta))
            if rescale:
                f = f / self.time_scaling
            std = torch.std(f, dim=0)
            if mean:
                f = f.mean(dim=0)

        else:
            f = torch.exp(self.get_prior_params)
            if rescale:
                f = f / self.time_scaling
            std = torch.zeros_like(f)

        f = f.view(-1, 1, self.n_proc, self.n_states, self.n_states) * self.diagonal_mask
        std = std.view(-1, 1, self.n_proc, self.n_states, self.n_states) * self.diagonal_mask

        if return_std:
            return f, std
        else:
            return f


class GaussianDecoder(BaseDecoder):
    """
    Reconstructs data from latent samples.
    Gaussian emission model, optionally with full covariance matrix
    """

    def __init__(self, data_dim, n_states, n_proc, layer_normalization, dropout, **kwargs):

        super(GaussianDecoder, self).__init__(
            data_dim, n_states, n_proc, layer_normalization, dropout
        )

        # latent samples can be one-hots
        one_hot_input = kwargs.get("one_hot_input", False)
        input_dim = n_proc * n_states if one_hot_input is True else n_proc
        self.index_dec_input = 1 if one_hot_input else 0

        self.z_as_mean = kwargs.get("z_as_mean", False)

        # covariance can be fixed and diagonal
        self.diag_cov = kwargs.get("diag_cov", False)
        fixed_diag_cov = kwargs.get("fixed_diag_cov", "None")
        self.fixed_diag_cov = fixed_diag_cov if fixed_diag_cov != "None" else None

        if self.diag_cov and self.z_as_mean:
            output_mlp_dim = self.data_dim
        elif self.diag_cov and not self.z_as_mean:
            output_mlp_dim = 2 * self.data_dim
        else:
            output_mlp_dim = 2 * self.data_dim + int(self.data_dim * (self.data_dim - 1) / 2)

        # set up MLP for reconstruction
        activation_mlp_decoder = kwargs.get("activation_mlp_decoder")
        layers_mlp_decoder = kwargs.get("layers_mlp_decoder")
        init_mlp_decoder = kwargs.get("init_mlp_decoder", None)

        self.get_decoder_param = MLP(
            input_dim,
            layers_mlp_decoder,
            output_mlp_dim,
            activation=activation_mlp_decoder,
            layer_normalization=layer_normalization,
            init_method=init_mlp_decoder,
            dropout=dropout,
        )

    def forward(self, z):
        """
        Returns mean and standard deviation as reconstructions based on latent samples.
        sample_param : (z, z_one_hot);
            z sample [B, T, n_proc, 1]
            z_one_hot [B,T,n_proc,n_states]
        return
        mean, sigma, predictions: [B, T, n_proc, D]
        """

        z_dec_input = z[self.index_dec_input]
        batch_size, seq_len, _, _ = z_dec_input.shape

        dec_param = self.get_decoder_param(z_dec_input.view(batch_size, seq_len, -1))

        # Learn Cholesky decomposition: cov^-1 = L * L^T
        # gather mean reconstruction and diagonal of cholesky decomposition of inverse of covariance matrix
        if self.z_as_mean:
            mean = z[0].flatten(start_dim=2)
            log_L_T_diag = (
                dec_param
                if self.fixed_diag_cov is None
                else np.log(1 / self.fixed_diag_cov) * torch.ones_like(mean)
            )
        else:
            mean, log_L_T_diag = (
                dec_param[..., : self.data_dim],
                dec_param[..., self.data_dim : 2 * self.data_dim],
            )
            if self.fixed_diag_cov != None:
                log_L_T_diag = np.log(1 / self.fixed_diag_cov) * torch.ones_like(mean)

        # assemble L^T only if needed
        if self.diag_cov:
            L_T = None
        else:
            if self.fixed_diag_cov is None:
                L_T_upper = dec_param[..., 2 * self.data_dim :]
            else:
                L_T_upper = torch.zeros_like(dec_param[..., 2 * self.data_dim :])

            L_T = torch.diag_embed(torch.exp(log_L_T_diag))

            # Add upper triangular, off-diagonal entries
            l_ind = torch.triu_indices(row=self.data_dim, col=self.data_dim, offset=1)
            L_T[:, :, l_ind[0], l_ind[1]] = L_T_upper

        return (mean, (L_T, log_L_T_diag))

    def get_reconstruction_loss(self, x_target, dec_param, q_obs, mask, return_batch_mean=True):
        """
        Returns the log likelihood for the gaussian model
        x_target: [B, T, data_dim]
        mask: None or [B, T, data_dim]
        dec_parameters --> mean, (L_T, log_l_T_diag): [B, T, data_dim], [B, T, data_dim, data_dim], [B, T, data_dim]
        """
        mean, (L_T, log_L_T_diag) = dec_param

        if mask is not None:
            mean = torch.where(mask == 1.0, mean, x_target)
            log_L_T_diag = torch.where(mask == 1.0, log_L_T_diag, torch.zeros_like(log_L_T_diag))

            mask = mask.unsqueeze(-1).expand(-1, -1, -1, self.data_dim)
            L_T = torch.where(mask == 1.0, L_T, torch.diag_embed(log_L_T_diag, dim1=-1, dim2=-2))

        if self.diag_cov:
            y = torch.exp(log_L_T_diag) * (x_target - mean)  # [B, T, data_dim]
        else:
            y = (L_T * (x_target - mean).unsqueeze(-2)).sum(dim=-1)  # [B, T, data_dim]

        rec_error = -1 / 2 * (y * y).sum(dim=-1)  # [B, T]
        det_error = log_L_T_diag.sum(dim=-1)  # [B, T]
        const_error = -1 / 2 * self.data_dim * np.log(2 * np.pi)

        ll = (rec_error + det_error + const_error).sum(dim=-1)  # [B]
        if return_batch_mean:
            return ll.mean()  # [1]
        else:
            return ll

    def sample(self, dec_param):
        """
        Returns sample of learned gaussian distribution.
        """
        mean, (L_T, log_L_T_diag) = dec_param
        eps = torch.randn_like(mean)

        if self.diag_cov:
            # log_L_T_diag is log(1/sigma)
            sample = mean + eps * 1 / torch.exp(log_L_T_diag)
        else:
            L = torch.inverse(L_T)
            sample = mean + (torch.matmul(L, eps.unsqueeze(-1))).squeeze(-1)

        return sample


class CategoricalDecoder(BaseDecoder):
    """
    Decoder for one-hot observations that uses Cross entropy.
    """

    def __init__(self, data_dim, n_states, n_proc, layer_normalization, dropout, **kwargs):
        super(CategoricalDecoder, self).__init__(
            data_dim, n_states, n_proc, layer_normalization, dropout
        )

        # if data_dim = n_proc * n_states we can also use log(q_obs) as logits
        self.latent_distribution_as_reconst = kwargs.get("latent_distribution_as_reconst", False)

        if self.latent_distribution_as_reconst:
            self.index_latent_type = 1
            self.get_decoder_param = lambda x: x
        else:
            # set up MLP for reconstruction
            activation_mlp_decoder = kwargs.get("activation_mlp_decoder")
            layers_mlp_decoder = kwargs.get("layers_mlp_decoder")
            init_mlp_decoder = kwargs.get("init_mlp_decoder", None)
            one_hot_input = kwargs.get("one_hot_input", False)
            input_dim = n_proc * n_states if one_hot_input is True else n_proc

            self.index_latent_type = 1 if one_hot_input else 0
            self.get_decoder_param = MLP(
                input_dim,
                layers_mlp_decoder,
                data_dim,
                activation=activation_mlp_decoder,
                layer_normalization=layer_normalization,
                init_method=init_mlp_decoder,
                dropout=dropout,
            )

        use_bce = kwargs.get("use_bce", False)  # BCE is better if n_proc > 1
        self.crit = nn.BCEWithLogitsLoss() if use_bce else nn.CrossEntropyLoss()

    def forward(self, z):
        """
        Returns logits based on latent samples. If latent_distribution_as_reconst is True, result is not used further.
        sample_param : (z, z_one_hot);
            z sample [B, T, n_proc, 1]
            z_one_hot [B,T,n_proc,n_states]
        return
        """
        z = z[self.index_latent_type]
        batch_size, seq_len, n_proc, _ = z.shape

        logits = self.get_decoder_param(z.view(batch_size, seq_len, -1))  # [B, T, data_dim]

        return logits

    def get_reconstruction_loss(self, x_target, dec_param, q_obs, mask):
        """
        Decoder compares learned q_obs to one-hot observations with Cross Entropy loss.
            x_target: [B, T, data_dim]
            mask: None or [B, T, data_dim]
            dec_parameters --> logits: [B, T, n_proc, n_states]
        """

        if self.latent_distribution_as_reconst is True:
            q_obs = q_obs.clamp(min=1e-6)
            q_obs = q_obs / q_obs.sum(dim=-1, keepdim=True)
            logits = torch.log(q_obs).flatten(start_dim=-2)  # [B,T, n_proc * n_states]
        else:
            logits = dec_param

        x_target = x_target.flatten(start_dim=0, end_dim=1)  # [B * T, data_dim]
        logits = logits.flatten(start_dim=0, end_dim=1)  # [B * T, data_dim]

        if mask is not None:
            mask = mask.flatten(start_dim=0, end_dim=1)  # [B * T, data_dim]
            logits = torch.where(mask == 1.0, logits, x_target)

        ll = -self.crit(logits, x_target)

        return ll

    def sample(self, dec_params):
        """
        Returns sample of learned categorical distribution.
        """

        if self.latent_distribution_as_reconst:
            sample = dec_params
        else:
            logits = dec_params
            m = torch.distributions.Categorical(logits=logits)
            sample = torch.nn.functional.one_hot(
                m.sample(), num_classes=self.data_dim
            )  # [B, T, data_dim]

        return sample

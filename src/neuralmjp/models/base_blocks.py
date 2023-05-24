import torch
from torch import nn as nn

from torchdiffeq import odeint, odeint_adjoint

from ..utils.helper import get_class_nonlinearity

# initialization methods
def init_normal(m):
    if type(m) in [nn.Linear]:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0)


def init_kai_normal(m):
    if type(m) in [nn.Linear]:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def init_xav_normal(m):
    if type(m) in [nn.Linear]:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def _build_layers(
    activation_fn,
    input_dim: int,
    layer_normalization: bool,
    layers: list,
    out_activation,
    output_dim: int,
    dropout=0.1,
) -> nn.Sequential:
    layer_sizes = [input_dim] + list(map(int, layers))
    layers = nn.Sequential()
    for i in range(len(layer_sizes) - 1):
        layers.add_module(f"layer {i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if layer_normalization:
            layers.add_module(f"layer norm {i}", nn.LayerNorm(layer_sizes[i + 1]))
        if dropout != 0.0:
            layers.add_module(f"dropout {i}", nn.Dropout(dropout))
        layers.add_module(f"activation {i}", activation_fn())
    layers.add_module("output layer", nn.Linear(layer_sizes[-1], output_dim))
    if out_activation is not None:
        out_activation_fn = get_class_nonlinearity(out_activation)
        layers.add_module("out activation", out_activation_fn())
    return layers


class MLP(nn.Module):
    """
    Custom MLP class configurable by yaml.
    """

    def __init__(
        self,
        input_dim,
        layers,
        output_dim,
        activation="LeakyReLU",
        out_activation=None,
        layer_normalization=True,
        init_method=None,
        dropout=0.1,
    ):
        super(MLP, self).__init__()
        activation_fn = get_class_nonlinearity(activation)
        self.data_dim = input_dim
        self.output_dim = output_dim
        self.layers = _build_layers(
            activation_fn,
            input_dim,
            layer_normalization,
            layers,
            out_activation,
            output_dim,
            dropout,
        ).float()

        if init_method == "kai_normal":
            self.layers.apply(init_kai_normal)
        elif init_method == "xav_normal":
            self.layers.apply(init_xav_normal)
        else:
            self.layers.apply(init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.layers(x)

    @property
    def device(self):
        return next(self.parameters()).device


class Block(nn.Module):
    """
    Defines device of module to initialize all tensors of module on the same device.
    """

    def __init__(self):
        super(Block, self).__init__()

    @property
    def device(self):
        return next(self.parameters()).device


class BlockODE(Block):
    """
    Specifies options for torchdiffeq.odeint by yaml.
    Simplifies calls to torchdiffeq.odeint.
    """

    def __init__(self, **kwargs):
        super(BlockODE, self).__init__()

        self.use_adjoint_method = kwargs.get("use_adjoint_method", False)
        self.solver_method = kwargs.get("solver_method", "dopri5")
        self.atol = kwargs.get("atol", 1e-2)
        self.rtol = kwargs.get("rtol", 1e-2)
        self.adjoint_atol = kwargs.get("adjoint_atol", 1e-2)
        self.adjoint_rtol = kwargs.get("adjoint_rtol", 1e-2)

    def _call_odeint(self, f, z_0, t, adjoint_params=None, norm=None):
        """
        Calls ode solver
        f: function to be integrated. Returns torch.Tensor
        x_0: initial conditions
        t: time
        adjoint_params: tuple of parameters added to adjoint system
        """
        if adjoint_params is None:
            adjoint_params = tuple(self.parameters())

        if self.use_adjoint_method:
            z = odeint_adjoint(
                f,
                z_0,
                t,
                adjoint_params=adjoint_params,
                method=self.solver_method,
                atol=self.atol,
                rtol=self.rtol,
                adjoint_rtol=self.adjoint_rtol,
                adjoint_atol=self.adjoint_atol,
            )

        else:
            z = odeint(
                f,
                z_0,
                t,
                method=self.solver_method,
                atol=self.atol,
                rtol=self.rtol,
            )

        return z


class BaseEncoder(BlockODE):
    """
    Base class for modules encoding observations into hidden states.
    Encoding defines initial condition and transition rates of posterior process.
    """

    def __init__(
        self,
        data_dim,
        n_states,
        n_proc,
        layer_normalization,
        dropout,
        max_obs_time,
        **kwargs,
    ):
        super(BaseEncoder, self).__init__(**kwargs)
        self.data_dim = data_dim  # data-dimension per process
        self.n_states = n_states  # number of states in latent space
        self.n_proc = n_proc  # number of processes in latent space
        self.layer_normalization = layer_normalization
        self.dropout = dropout
        self.max_obs_time = max_obs_time  # maximum observation time in data

        self.time_data_rep_dim = None  # dimension of time-data representation h(t) (g = g(h(t))
        self.me_adjoint_params = None  # tuple of parameters that influence the master eq (for odeint_adjoint of master eq)

    def get_me_adjoint_params(self):
        """
        Return model-parameters that influence self.time_data_rep.
        self.me_adjoint_params has to be specified during initialization.
        """
        return self.me_adjoint_params

    def get_time_data_rep(self):
        """
        Returns the time-data representation method.
        Used for transition rates of posterior process.
        """
        return self.time_data_rep

    def get_time_data_rep_dim(self):
        """
        Return output-dimension of self.time_data_rep.
        Used in construction of MLPs for mean-field rates g.
        self.time_data_rep_dim has to be specified during initialization.
        """
        return self.time_data_rep_dim

    def time_data_rep(self, t):
        """
        Continuous time embedding of observations at time t with dimension time_data_rep_dim.
        t: [B, *, 1]
        return: [B, *, time_data_rep_dim]   --- time-data embedding at t
        """

    def forward(self, x):
        """
        Returns initial distribution and sets up time_data_rep if necessary.
        x: [B, T, data_dim + 2]          --- observations, time and time-differences
        return: [B, n_proc, n_states]    --- initial posterior distribution
        """


class BaseLatentProcess(Block):
    """
    Specifies methods to be implemented and variables to be defined that are necessary for implementation of
    each specific latent process.
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
        **kwargs,
    ):
        super(BaseLatentProcess, self).__init__()
        self.n_proc = n_proc  # number of latent processes
        self.n_states = n_states  # number of latent states

        # function returning time-data representation h(t) a time t
        self.time_data_rep = time_data_rep

        self.time_data_rep_dim = time_data_rep_dim  # dimension of h(t); to be used for gs
        self.layer_normalization = layer_normalization
        self.dropout = dropout
        self.time_scaling = time_scaling  # time scaling depending on data

        # Prior parameters of model
        self.use_generative_prior_params = kwargs.get("use_generative_prior_params", True)

        # prior parameters as torch.nn.Parameters
        if not self.use_generative_prior_params:
            self.get_prior_params = nn.Parameter(torch.randn(n_prior_params), requires_grad=True)
            torch.nn.init.normal_(self.get_prior_params, mean=0.0, std=0.01)

        # prior parameters by generative model
        else:
            self.input_dim_prior_params_mlp = kwargs.get("input_dim_prior_params_mlp", 64)
            layers_prior_params = kwargs.get("layers_prior_params", (32, 32))
            activation_prior_params = kwargs.get("activation_prior_params", "Tanh")
            init_prior_params = kwargs.get("init_prior_params", "None")
            self.input_std_prior_params_mlp = kwargs.get("input_std_prior_params_mlp", 0.1)
            self.get_prior_params = MLP(
                self.input_dim_prior_params_mlp,
                layers_prior_params,
                n_prior_params,
                activation=activation_prior_params,
                init_method=init_prior_params,
                layer_normalization=layer_normalization,
                dropout=dropout,
            )

        # tuple of parameters that influence the master eq (for odeint_adjoint of master eq)
        self.me_adjoint_params = None

    def get_me_adjoint_params(self):
        """
        Return model-parameters that influence self.get_rates.
        self.me_adjoint_params has to be specified during intitialisation.
        """
        return self.me_adjoint_params

    def get_dqdt(self, q, t):
        """
        Returns the derivative of the (local/aggregated) distribution over states
        for each process
        q: [B, *, n_proc, n_states]
        t: [B, *, 1]
        return: [B, *, n_proc, n_states]
        """

    def get_kl_integrand(self, t, q):
        """
        Computes integrand of Kullback Leibler distance of prior and posterior process.
        t: [B, T, 1]                            --- quadrature times
        q: [B, T, num_proc, n_states]           --- (mean-field) aggregated probability over states
        returns: [B, T]
        """

    def prior_params(self, num_samples=1, rescale=True, mean=False):
        """
        Returns prior parameters.
        num_samples: if self.use_generative_prior_params, sample num_samples prior params
        rescale: if True, return prior parameters rescaled to original time scale
        mean: returns mean parameters over num_samples
        """

    @staticmethod
    def add_stats(stats):
        """
        Add entries to loss stats dictionary that are specific to latent process.
        Override method if necessary.
        """
        return stats


class BaseDecoder(Block):
    """
    Base class for emission models.
    """

    def __init__(self, data_dim, n_states, n_proc, layer_normalization, dropout):
        super(BaseDecoder, self).__init__()

        self.data_dim = data_dim
        self.n_states = n_states
        self.n_proc = n_proc
        self.layer_normalization = layer_normalization
        self.dropout = dropout

    def forward(self, sample_param):
        """
        Returns reconstruction based on latent samples
        sample_param : (z, z_one_hot);
            z sample [B, T, n_proc, 1]
            z_one_hot [B,T,n_proc,n_states]
        return
        tuple(dec_param), where dec_param is passed to self.get_reconstruction_loss
        """

    def get_reconstruction_loss(self, x_target, dec_param, q_obs, mask):
        """
        Return reconstruction loss
        x_target: [B, T, data_dim]
        dec_parameters: *
        q_obs: [B, T, n_proc, n_states]
        mask: None or [B, T, data_dim]
        return: [1] --- log-likelihood
        """

    def sample(self, dec_param):
        """
        Returns sample of decoder
        dec_param: output of forward method
        return: [B, T, n_proc, D]
        """

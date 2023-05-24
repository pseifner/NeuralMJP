import os
import numpy as np
import scipy.optimize
import scipy.stats

import torch
import math

import click
from pathlib import Path
import yaml

from neuralmjp import data_path


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file",
)
def main(cfg_path: Path):
    with open(cfg_path, "rb") as f:
        params = yaml.full_load(f)

    clazz = globals()[params["data_model"]["name"]]
    instance = clazz(**params)

    instance.generate_data()


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def save_data(data, data_description, data_path, params):
    """
    Save generated data and simulation parameters.
    Data: list of generated data
    Data_description: list of labels for data
    """
    # Save data
    print("Saving data in directory:", data_path)
    os.makedirs(data_path, exist_ok=True)
    file_names = [s + ".npy" for s in data_description]

    for i in range(len(data)):
        combined_data_path = os.path.join(data_path, file_names[i])
        remove_file_if_exists(combined_data_path)
        with open(combined_data_path, "wb") as f:
            np.save(f, data[i])

    # Save config params
    yaml_path = os.path.join(data_path, "simulation_parameters.yaml")
    remove_file_if_exists(yaml_path)
    with open(yaml_path, "w") as out:
        yaml.dump(params, out, default_flow_style=False)


# Markov Processes
class MP:
    """
    Generate trajectories of either Markov Jump Process or Markov Chain.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        simul_params = kwargs["simulation"]

        self.model = simul_params.get("model", "MJP")
        self.seed = simul_params.get("seed", 0)

        # Number of time series to generate and length of each time series
        self.B = simul_params.get("B", 5000)
        self.T = simul_params.get("T", 50)

        # Standard deviation for additive gaussian noise
        self.gaussian_std = simul_params.get("gaussian_std", 1)

        # Additive term in Opper Equation (13) for additive exponential noise
        self.opper_correction_term = simul_params.get("opper_correction_term", 0.000001)

        # Additional possible states for opper discrete noise above and below min and max observed state
        self.opper_additional_range = simul_params.get("opper_additional_range", 0)

        if self.model == "MJP":
            MJP_config = simul_params["MJP_config"]
            # Generate trajectories up to time t_max
            self.t_max = MJP_config.get("t_max", 1)
            self.generate_prediction_set = MJP_config.get("generate_prediction_set", False)
            if self.generate_prediction_set:
                self.t_max = 2 * self.t_max
                self.T = 2 * self.T
            self.share_observation_times = MJP_config.get("share_observation_times", True)
            self.regular_grid = MJP_config.get("regular_grid", False)
            self.load_non_pred_grid = MJP_config.get("load_non_pred_grid", None)

        self.num_processes = None

        # Path for saving generated data
        use_default_dir = simul_params.get("use_default_dir", True)
        if use_default_dir is True:
            self.data_path = os.path.join(data_path, kwargs["name"])
        else:
            self.data_path = os.path.join(
                os.getcwd(), simul_params["non_default_dir"], kwargs["name"]
            )

    def generate_data(self):
        """
        Generate process trajectories, observe it at (randomly drawn) time-points, add noise and save data.
        """
        # Sample complete trajectories
        process_trajectories = self.generate_true_process()
        self.num_processes = process_trajectories.shape[-1] - 1
        print("Process trajectories generated.")

        # Generate observations from complete trajectories at random times
        if self.model == "MJP":
            obs_trajectories = self.generate_observations(process_trajectories)
            print("Observations generated.")
        elif self.model == "MC":
            obs_trajectories = process_trajectories

        # Apply several noises to observations
        obs_exp_noise = self.add_exponential_noise(obs_trajectories)
        print("Observations with exponential noise generated.")
        obs_opper_exp_noise = self.add_opper_exponential_noise(obs_trajectories)
        print("Observations with exponential noise and opper correction term generated.")
        obs_gaussian_noise = self.add_gaussian_noise(obs_trajectories)
        print("Observations with gaussian noise generated.")
        obs_discrete_opper_noise = self.discrete_opper_noise(obs_trajectories)
        print("Observations with discrete opper noise generated.")

        # Save data
        data_description = [
            "true_obs",
            "exponential_noise",
            "opper_corrected_exponential_noise",
            "gaussian_noise",
            "discrete_opper_noise",
            "true_trajectories",
        ]

        if self.model == "MJP":
            data = [
                obs_trajectories,
                obs_exp_noise,
                obs_opper_exp_noise,
                obs_gaussian_noise,
                obs_discrete_opper_noise,
                process_trajectories,
            ]
        if self.model == "MC":
            # drop time, as that is not part of MC model
            data = [
                obs_trajectories[..., 1:],
                obs_exp_noise[..., 1:],
                obs_opper_exp_noise[..., 1:],
                obs_gaussian_noise[..., 1:],
                obs_discrete_opper_noise[..., 1:],
            ]

        save_data(data, data_description, self.data_path, self.params)

    def generate_true_process(self):
        """
        Returns samples of process up to time t_max.
        Return shape: [B,Z,1 + num_processes], where Z is number of transitions such that every sample has reached t_max.
        """
        process_trajectories = torch.cat(
            [torch.zeros(self.B, 1), self.get_initial_states()], dim=-1
        ).unsqueeze(1)
        # process_trajectories.shape = [B,1,1+num_processes]

        if self.model == "MJP":
            while (process_trajectories[:, -1, 0] < self.t_max).any():
                time, state = (
                    process_trajectories[:, -1, 0].unsqueeze(-1),
                    process_trajectories[:, -1, 1:],
                )
                update = self.update_process(time, state)
                process_trajectories = torch.cat([process_trajectories, update.unsqueeze(1)], dim=1)

        elif self.model == "MC":
            for i in range(self.T - 1):
                time, state = (
                    process_trajectories[:, -1, 0].unsqueeze(-1),
                    process_trajectories[:, -1, 1:],
                )
                update = self.update_process(time, state)
                process_trajectories = torch.cat([process_trajectories, update.unsqueeze(1)], dim=1)

        return process_trajectories

    def generate_observations(self, process_trajectories):
        """
        Observe process trajectory at T (randomly drawn) observation times.
        Process_trajectories shape: [B, Z, 1 + num_processes], where Z is number of transitions needed for all B trajectories to reach t_max
        Return shape: [B, T, 1 + num_processes]
        """
        num_processes = process_trajectories.shape[-1] - 1

        trans_times = process_trajectories[:, :, 0]

        # sample observation times
        if self.generate_prediction_set:
            m1 = torch.distributions.Uniform(low=0, high=self.t_max / 2)
            m2 = torch.distributions.Uniform(low=self.t_max / 2, high=self.t_max)
        else:
            m = torch.distributions.Uniform(low=0, high=self.t_max)

        if self.share_observation_times == True:
            if self.regular_grid == True:
                obs_times = torch.linspace(start=0, end=self.t_max, steps=self.T).unsqueeze(0)
                obs_times = obs_times.expand(self.B, self.T)
            else:
                if self.generate_prediction_set:
                    if self.load_non_pred_grid is None:
                        obs_times_a = m1.sample(sample_shape=(1, int(self.T / 2)))
                    else:
                        with open(self.load_non_pred_grid, "rb") as f:
                            obs = np.load(f)
                            obs = torch.from_numpy(obs)
                            obs_times_a = obs[0, :, 0].reshape(1, -1)
                    obs_times_b = m2.sample(sample_shape=(1, int(self.T / 2)))
                    obs_times = torch.cat([obs_times_a, obs_times_b], dim=-1)
                else:
                    obs_times = m.sample(sample_shape=(1, self.T))
                obs_times = obs_times.expand(self.B, self.T)
        else:
            if self.generate_prediction_set:
                obs_times_a, obs_times_b = m1.sample(
                    sample_shape=(self.B, int(self.T / 2))
                ), m2.sample(sample_shape=(self.B, int(self.T / 2)))
                obs_times = torch.cat([obs_times_a, obs_times_b], dim=-1)
            else:
                obs_times = m.sample(sample_shape=(self.B, self.T))

        obs_times, _ = obs_times.sort(dim=-1)

        # gather observations
        all_obs = []
        for i in range(self.T):
            time = obs_times[:, i].unsqueeze(-1).expand(-1, trans_times.shape[-1])
            indices = (trans_times <= time).count_nonzero(dim=1) - 1
            indices = indices.view(-1, 1, 1).expand(-1, -1, num_processes + 1)
            obs = torch.gather(input=process_trajectories, index=indices, dim=1)
            all_obs.append(obs)

        obs_trajectories = torch.cat(all_obs, dim=1)

        # replace times from raw data with observation times
        obs_trajectories[:, :, 0].copy_(obs_times)

        return obs_trajectories

    def get_initial_states(self):
        """
        Return initial states for all B trajectories.
        Has to be specified for each process individually.
        Return shape: [B, num_processes]
        """

    def get_trans_targets(self, state):
        """
        Return all possible states the process can transition to from a given state.
        Has to be specified for each process individually.
        Return shape: [B, X, num_processes] where X is number of possible transitions of process
        """

    def get_trans_rates(self, state):
        """
        Return current transition rates, given state of process. Shape and entries have to correspond with output of get_trans_targets.
        Has to be specified for each process individually.
        Return shape: [B, X]
        """

    def update_process(self, time, state):
        """
        Simulate jump in state and waiting-time of process
        Rates shape: [B, X]
        Targets shape: [B, X, num_processes]
        Time shape: [B, 1]
        Return shape: [B, 1 + num_processes]
        """
        proc_dim = state.shape[-1]
        # get current transition-rates of process
        rates = self.get_trans_rates(state)
        targets = self.get_trans_targets(state)

        # update state
        trans_probs = rates / (rates.sum(dim=-1, keepdims=True))
        m = torch.distributions.Categorical(probs=trans_probs)
        indices = m.sample().view(self.B, 1, 1).expand(self.B, 1, proc_dim)

        new_state = torch.gather(input=targets, dim=1, index=indices.to(torch.int64)).squeeze(1)

        # update time
        m = torch.distributions.Exponential(rate=rates.sum(dim=-1, keepdims=True))
        t_wait = m.sample()
        new_time = time + t_wait

        combined = torch.cat([new_time, new_state], dim=1)

        return combined

    def add_exponential_noise(self, obs_trajectories):
        """
        Add (additive) exponential noise to observations.
        Obs_trajectories shape: [B, T, 1 + num_processes]
        Return shape: [B, T, 1 + num_processes]
        """
        m = torch.distributions.Uniform(low=0, high=1)
        u = m.sample(sample_shape=(self.B, self.T, self.num_processes))
        val_noise = -torch.log(1 - u * math.log(2))

        noisy_trajectories = obs_trajectories + torch.cat(
            [torch.zeros(self.B, self.T, 1), val_noise], dim=-1
        )

        return noisy_trajectories

    def add_opper_exponential_noise(self, obs_trajectories):
        """
        Add (additive) exponential noise to observations, where the exponential distribution has an added correction term from Opper.
        Obs_trajectories shape: [B, T, 1 + num_processes]
        Return shape: [B, T, 1 + num_processes]
        """
        # No analytical inverse CDF due to the added correction term
        # Invert CDF numerially at uniform samples using Newton method
        m = torch.distributions.Uniform(low=0, high=1)
        u = m.sample(sample_shape=(self.B, self.T, self.num_processes))

        u = u.numpy()
        x = np.zeros_like(u)
        x = scipy.optimize.newton(self.exp_cum_dist, x, args=(u,))

        x = torch.from_numpy(x)

        noisy_trajectories = obs_trajectories + torch.cat(
            [torch.zeros(self.B, self.T, 1), x], dim=-1
        )

        return noisy_trajectories

    def exp_cum_dist(self, x, u):
        return self.opper_correction_term * x + (1 - np.power(2, -x)) / math.log(2) - u

    def add_gaussian_noise(self, obs_trajectories):
        """
        Add (additive) gaussian noise to observations.
        Obs_trajectories shape: [B, T, 1 + num_processes]
        Return shape: [B, T, 1 + num_processes]
        """
        num_processes = obs_trajectories.shape[-1] - 1
        noisy_trajectories = obs_trajectories + torch.cat(
            [
                torch.zeros(self.B, self.T, 1),
                self.gaussian_std * torch.randn((self.B, self.T, num_processes)),
            ],
            dim=-1,
        )

        return noisy_trajectories

    def discrete_opper_noise(self, obs_trajectories):
        """
        Add discrete opper noise from equation (13).
        Obs_trajectories shape: [B, T, 1 + num_processes]
        Return shape: [B, T, 1 + num_processes]
        """
        # Minimal and maximal observed states
        obs_states = obs_trajectories[..., 1:]
        min_obs, max_obs = torch.min(obs_states), torch.max(obs_states)

        # Range of possible noisy states: add additional states below min and above max
        possible_states = torch.arange(
            start=min_obs, end=max_obs + self.opper_additional_range + 1, step=1
        ).view(1, 1, 1, -1)

        # Set up distribution described in Opper Equation (13)
        p = (
            1 / torch.pow(2, torch.abs(possible_states - obs_states.unsqueeze(-1)))
            + self.opper_correction_term
        )
        p = p / p.sum(dim=-1, keepdims=True)
        c = torch.distributions.Categorical(p)

        # Torch sample only returns indices; convert indices to samples of states
        sampled_states = c.sample() + (min_obs - self.opper_additional_range)

        noisy_trajectories = torch.cat(
            [obs_trajectories[..., 0].unsqueeze(-1), sampled_states], dim=-1
        )

        return noisy_trajectories


class LV(MP):
    """
    Lotka-Volterra model
    """

    def __init__(self, **kwargs):
        super(LV, self).__init__(**kwargs)
        self.model_params = kwargs["data_model"]["args"]

        # Parameters of LV
        self.alpha = self.model_params.get("alpha")
        self.beta = self.model_params.get("beta")
        self.gamma = self.model_params.get("gamma")
        self.delta = self.model_params.get("delta")

        # Small constant rate for transitions 0->1
        self.min_up_rate = self.model_params.get("min_up_rate", 0.000001)

        # For random initialization of time series
        self.random_init = self.model_params.get("random_init", False)
        self.init_lower_bound = self.model_params.get("init_lower_bound", 5)
        self.init_upper_bound = self.model_params.get("init_upper_bound", 25)

        # If initialization is not random, initialize every sequence in the same state
        self.init_state = torch.tensor(kwargs.get("init_state", [19, 7]))

    def get_initial_states(self):
        """
        Return initial states for all B trajectories.
        Return shape: [B, num_processes]
        """
        if self.random_init is True:
            return torch.randint(
                low=self.init_lower_bound, high=self.init_upper_bound, size=(self.B, 2)
            )
        else:
            return self.init_state.unsqueeze(0).expand(self.B, 2)

    def get_trans_targets(self, state):
        """
        Return all possible states the process can transition to from a given state.
        Return shape: [B, X, num_processes] where X is number of possible transitions of process
        """
        relative_trans = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])

        return state.unsqueeze(1) + relative_trans

    def get_trans_rates(self, state):
        """
        Return current transition rates, given state of process. Shape and entries have to correspond with output of get_trans_targets.
        Return shape: [B, X]
        """
        x, y = state[:, 0].unsqueeze(-1), state[:, 1].unsqueeze(-1)

        rates = torch.cat(
            [self.alpha * x, self.beta * x * y, self.delta * x * y, self.gamma * y], dim=-1
        )
        # rates.shape = [B,4]

        # clamp rates for jumping up; this is what Opper does as well
        rates[:, 0] = rates[:, 0].clamp(self.min_up_rate)
        rates[:, 2] = rates[:, 2].clamp(self.min_up_rate)

        return rates


class DFR(MP):
    """
    Discrete flashing ratchet model
    Equations from: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.150607
    """

    def __init__(self, **kwargs):
        super(DFR, self).__init__(**kwargs)
        self.model_params = kwargs["data_model"]["args"]

        T = self.model_params["Temp"]
        self.beta = 1 / T
        self.r = self.model_params["r"]
        self.rates_when_potential_on = self.model_params["rates_when_potential_on"]

        V = self.model_params["V"]

        self.B = kwargs["simulation"]["B"]
        self.V = torch.tensor([0 * V, 1 * V, 2 * V])

        # build transition rate matrix K, where k_ij = trans rate from i to j (in all_states tensor)
        # will multiply K from right onto identified_states

        # rates for changes in x when eta == 0
        V_j = self.V.unsqueeze(0)
        V_i = self.V.unsqueeze(1)
        rates_x_eta_0 = torch.exp(-self.beta / 2 * (V_j - V_i)).fill_diagonal_(0)

        # rates for changes in x when eta == 1
        rates_x_eta_1 = (self.rates_when_potential_on * torch.ones((3, 3))).fill_diagonal_(0)

        # rates for changes in eta
        rates_eta = self.r * torch.eye(3)

        self.K = (
            torch.cat(
                [
                    torch.cat([rates_x_eta_0, rates_eta], dim=1),
                    torch.cat([rates_eta, rates_x_eta_1], dim=1),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .expand(self.B, 6, 6)
        )
        print("Matrix with transition rates K:\n", self.K[0])

        # Options for state dependent gaussian noise
        self.state_dependent_std = kwargs["simulation"].get(
            "state_dependent_std", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )

        # Option for projection of data
        self.proj_dim = kwargs["simulation"].get("proj_dim", 2)
        self.proj_gaussian_std = kwargs["simulation"].get("proj_gaussian_std", 0.1)

    def generate_data(self):
        """
        Additional types of noise for DFR model
        """
        super().generate_data()

        obs_trajectories = np.load(os.path.join(self.data_path, "true_obs.npy"))
        obs_state_gaussian_noise = self.add_state_dependent_gaussian_noise(obs_trajectories)
        print("Observations with state dependent gaussian noise generated")
        proj_trajectories = self.add_random_projection(obs_trajectories)
        print("Observations with random projection generated")

        # Save additional data
        data = [obs_state_gaussian_noise, proj_trajectories]
        data_description = ["gaussian_noise_state_dep", "projected_obs"]
        save_data(data, data_description, self.data_path, self.params)

    def add_state_dependent_gaussian_noise(self, obs_trajectories):
        """
        Add gaussian noise where std depends on state.
        """
        obs_times, obs = obs_trajectories[:, :, 0], obs_trajectories[:, :, 1:]

        noisy_trajectory = obs
        white_noise = np.random.randn(*noisy_trajectory.shape)
        for i in range(6):
            noisy_trajectory = (
                noisy_trajectory
                + (noisy_trajectory == i) * self.state_dependent_std[i] * white_noise
            )

        noisy_trajectory = np.concatenate([obs_times[:, :, None], noisy_trajectory], axis=-1)
        return noisy_trajectory

    def add_random_projection(self, obs_trajectories):
        """
        Project data to high dimensional space and add noise there.
        """
        obs_times, flat_data = obs_trajectories[:, :, 0], obs_trajectories[:, :, 1]

        # convert data to one hot vectors
        flat_data = flat_data[:, :, None]
        B, T, _ = flat_data.shape

        flat_data = flat_data.repeat(6, axis=-1)
        all_states = (
            np.array([0, 1, 2, 3, 4, 5]).reshape(1, 1, 6).repeat(B, axis=0).repeat(T, axis=1)
        )

        one_hots = (flat_data == all_states).astype(int)  # [B, T, 6]

        # sample and apply random projection matrix
        W_mu = np.random.randn(6, self.proj_dim)[None, None, :, :]
        one_hots = one_hots[:, :, None, :]
        z_mu = np.matmul(one_hots, W_mu).squeeze(axis=2)

        # add gaussian noise to projected data
        white_noise = np.random.randn(*z_mu.shape)
        proj_data = z_mu + self.proj_gaussian_std * white_noise

        # reattach observation times
        proj_data = np.concatenate([obs_times[:, :, None], proj_data], axis=2)

        # save projection matrix for reference
        save_data([W_mu], ["proj_matrix"], self.data_path, self.params)

        return proj_data

    def get_initial_states(self):
        """
        Return initial states for all B trajectories.
        Return shape: [B, num_processes]
        """
        # index = torch.randint(low=0, high=6, size=(self.B,1), dtype=torch.int64)

        # Eq. dist for V=b=r=1
        d = torch.tensor([0.30119, 0.13654, 0.06227, 0.2003, 0.15914, 0.14057])
        m = torch.distributions.Categorical(d)
        index = m.sample(sample_shape=(self.B, 1))
        return index

    def get_trans_targets(self, state):
        """
        Return all possible states the process can transition to from a given state.
        State shape: [B, num_processes]
        Return shape: [B, X, num_processes] where X is number of possible transitions of process
        """
        all_states = (
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).view(1, 6, 1).expand(self.B, -1, -1)
        )
        return all_states

    def get_trans_rates(self, state):
        """
        Return current transition rates, given state of process. Shape and entries have to correspond with output of get_trans_targets.
        State shape: [B, num_processes]
        Return shape: [B, X]
        """
        # state = state.unsqueeze(1).expand(self.B, 6, 2)
        state = state.unsqueeze(1).expand(self.B, 6, 1)

        all_states = self.get_trans_targets(state)

        # identify position of current state in all_states tensor
        identified_state = torch.all(state == all_states, dim=-1).to(torch.float)
        # identified_state has one 1 in last dimension, at the index the state has in all_states

        # multiply transition rate matrix from right to extract all transition rates for jumps from state
        rates = torch.bmm(identified_state.unsqueeze(-2), self.K).squeeze(-2)

        return rates


class BrownianDynamics:
    """
    Generate trajectories of Brownian Dynamics.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        simul_params = kwargs["simulation"]

        # Simulating brownian dynamics
        self.del_t = simul_params.get("del_t", 0.01)
        self.D = simul_params.get("D", 1)
        self.kT = simul_params.get("kT", 1)
        self.burn_in_steps = simul_params.get("burn_in_steps", 1000)
        self.n_skip = simul_params.get("n_skip", 1)

        # Number of batches and length of each time series
        self.B = simul_params.get("B", 10000)
        self.T = simul_params.get("T", 100)

        self.traj_length = self.T + self.burn_in_steps

        # Path for saving generated data
        if simul_params["use_default_dir"] == True:
            self.data_path = os.path.join(data_path, kwargs["name"])
        else:
            self.data_path = os.path.join(
                os.getcwd(), simul_params["non_default_dir"], kwargs["name"]
            )

    def generate_data(self):
        """
        Generate and save trajectories.
        """
        trajectories = self.generate_trajectories()  # [B, traj_length, 1 + D]

        # remove burn_in
        trajectories = trajectories[:, self.burn_in_steps :, :]

        # adjust observation time so each trajectory starts at time 0
        trajectories[:, :, 0] = trajectories[:, :, 0] - trajectories[:, 0, 0].unsqueeze(1)

        data = [trajectories]
        data_description = ["true_obs"]

        save_data(data, data_description, self.data_path, self.params)

    def du(self, x):
        """
        Potential energy.
        Has to be implemented for each Brownian dynamic.
        """

    def get_initial_value(self):
        """
        Initial value of each trajectory.
        Has to be implemented for each Brownian dynamic.
        """

    def step(self, x):
        """
        One step in brownian dynamics.
        """
        w = torch.randn_like(x)
        return x - self.del_t * self.du(x) / self.kT + math.sqrt(2 * self.del_t * self.D) * w

    def generate_trajectories(self):
        """
        Generate trajectories of brownian dynamics including burn in time.
        """
        # generate trajectory and record every n_skip value
        x = self.get_initial_value()  # [B, D]
        trajectories = [x]

        steps = self.traj_length * self.n_skip
        steps = steps if self.n_skip == 1 else steps - 1
        for i in range(1, steps):
            x = self.step(x)
            if i % self.n_skip == 0:
                trajectories.append(x)

        trajectories = torch.stack(trajectories, dim=1)  # [B, traj_length, D]

        # add observation times
        t_obs = (
            (torch.arange(start=0, end=self.traj_length) * self.n_skip * self.del_t)
            .view(1, -1, 1)
            .expand(self.B, self.traj_length, 1)
        )

        trajectories = torch.cat([t_obs, trajectories], dim=-1)  # [B, traj_length, 1 + D]

        return trajectories


class ToyProteinFolding(BrownianDynamics):
    """
    Generate toy dataset of protein folding model from VAMPnets.
    """

    def __init__(self, **kwargs):
        super(ToyProteinFolding, self).__init__(**kwargs)
        model_params = kwargs["data_model"]["args"]

        self.use_potential_from_code = model_params.get("use_potential_from_code", True)

    def du(self, x):
        """Return potential energy from protein folding model in VAMPnets."""
        if self.use_potential_from_code is True:
            # du from VAMPnets code (I think there is a computation error in their gradient)
            r = torch.linalg.norm(x, dim=-1, keepdim=True) * torch.ones_like(x)
            du = torch.where(r < 3, -5 * (r - 3) * x / r, (1.5 * (r - 3) - 2) * x / r)
        else:
            # alternative du from potential in VAMPnets paper
            r = torch.linalg.norm(x, dim=-1, keepdim=True) * torch.ones_like(x)
            du = torch.where(
                r < 3, -5 * (r - 3) * x / r, 1.5 * ((r - 3) ** 2) * x / r - 2 * (r - 3) * x / r
            )
        return du

    def get_initial_value(self):
        """Initial value used in VAMPnets code"""
        x0 = 2 * (torch.randn(self.B, 5) - 0.5)
        return x0


class ADP:
    """
    Preprocesses Alanine Dipeptide Protein data
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        gen_params = kwargs["generation"]

        self.T = gen_params.get("T", 100)  # Number of observations in each timeseries

        self.test_size = gen_params.get("test_size", 100)  # amount of test time series

        self.selected_atoms = gen_params.get(
            "selected_atoms", [4, 6, 8, 10, 14, 16]
        )  # indices of selected atoms
        self.generation_dt = gen_params.get(
            "generation_dt", 1
        )  # dt data was originally generated with

        self.gmvae_kurtosis_threshold = gen_params.get("gmvae_kurtosis_threshold", -0.03)

        self.raw_data_path = gen_params.get(
            "raw_data_path"
        )  # path to complete all atoms trajectory

        # Path for saving generated data
        use_default_dir = gen_params.get("use_default_dir", True)
        if use_default_dir is True:
            self.data_path = os.path.join(data_path, kwargs["name"])
        else:
            self.data_path = os.path.join(
                os.getcwd(), gen_params["non_default_dir"], kwargs["name"]
            )

    def generate_data(self):
        raw_data = torch.from_numpy(np.load(self.raw_data_path))  # [*, 22, 3]

        selected_atoms = raw_data[:, self.selected_atoms].flatten(start_dim=1)  # [*, *, 3]
        ramach_traj = self.dihedral_transform(raw_data)  # [*, 2]
        gmvae_traj, sin_cos_ramach_traj = self.gmvae_transform(raw_data)  # [*, 24], [*, 4]
        standard_pairwise_distance = self.pwise_dist(raw_data)  # [*, _]

        # create batches
        traj_len, D = selected_atoms.shape
        B = int(traj_len / self.T)  # number of batches in total
        pred_indices = (
            np.random.choice(np.arange(B / 2, step=1), size=self.test_size, replace=False).astype(
                int
            )
            * 2
        )  # first half of prediction indices

        selected_atoms, selected_atoms_test = self.create_batches(selected_atoms, pred_indices)
        ramach_traj, ramach_traj_test = self.create_batches(ramach_traj, pred_indices)
        gmvae_traj, gmvae_traj_test = self.create_batches(gmvae_traj, pred_indices)
        sin_cos_ramach_traj, sin_cos_ramach_traj_test = self.create_batches(
            sin_cos_ramach_traj, pred_indices
        )
        standard_pairwise_distance, standard_pairwise_distance_test = self.create_batches(
            standard_pairwise_distance, pred_indices
        )

        # Save data
        data_description = [
            "selected_atoms",
            "ramach_angles",
            "gmvae_transform",
            "cos_sin_ramach_angles",
            "stand_pwise_dist",
            "selected_atoms_test",
            "ramach_angles_test",
            "gmvae_transform_test",
            "cos_sin_ramach_angles_test",
            "stand_pwise_dist_test",
        ]
        data = [
            selected_atoms,
            ramach_traj,
            gmvae_traj,
            sin_cos_ramach_traj,
            standard_pairwise_distance,
            selected_atoms_test,
            ramach_traj_test,
            gmvae_traj_test,
            sin_cos_ramach_traj_test,
            standard_pairwise_distance_test,
        ]

        save_data(data, data_description, self.data_path, self.params)

    def create_batches(self, data, pred_indices):
        """
        data: [*, D]
        pred_indices: [test_size]
        return: [B, T, 1+x], [test_size, 2*T, 1+x]
        """
        traj_len, D = data.shape

        # drop leftover data after batching
        l = int(traj_len / self.T) * self.T
        data = data[:l]

        # create batches
        data = data.reshape(-1, self.T, D)

        # split off test series of twice the length
        pred_indices_ = pred_indices + 1

        test, test_ = np.take(data, pred_indices, axis=0), np.take(
            data, pred_indices_, axis=0
        )  # [test_size, T, D], [test_size, T, D]
        test = np.concatenate([test, test_], axis=-2)  # [test_size, 2*T, D]

        # delete test data from data
        data = np.delete(data, np.concatenate([pred_indices, pred_indices_]), axis=0)

        # Append observation times
        obs_times_data = np.arange(start=0, stop=self.T, step=1) * self.generation_dt
        obs_times_data = np.broadcast_to(obs_times_data[None, :, None], (data.shape[0], self.T, 1))
        obs_times_test = np.arange(start=0, stop=2 * self.T, step=1) * self.generation_dt
        obs_times_test = np.broadcast_to(
            obs_times_test[None, :, None], (test.shape[0], 2 * self.T, 1)
        )

        data = np.concatenate([obs_times_data, data], axis=-1)
        test_data = np.concatenate([obs_times_test, test], axis=-1)

        return data.astype(float), test_data.astype(float)

    def dihedral_transform(self, raw_data):
        indices_psi = [6, 8, 14, 16]
        indices_phi = [4, 6, 8, 14]

        x = np.array(raw_data)
        psi, phi = self.compute_dihedral_angle(x, indices_psi), self.compute_dihedral_angle(
            x, indices_phi
        )
        angles = np.concatenate([psi, phi], axis=-1)  # [*, 2]

        return angles

    def gmvae_transform(self, raw_data):
        raw_data = np.array(raw_data)

        # select heavy atoms
        indices = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
        heavy_atoms = raw_data[:, indices, :]

        # transform into ramachandran angles
        indices_psi = [6, 8, 14, 16]
        indices_phi = [4, 6, 8, 14]
        psi, phi = self.compute_dihedral_angle(raw_data, indices_psi), self.compute_dihedral_angle(
            raw_data, indices_phi
        )
        psi, phi = np.radians(psi), np.radians(phi)

        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        ram_traj = np.concatenate([cos_psi, sin_psi, cos_phi, sin_phi], axis=-1)  # [T, 4]

        # Pairwise difference
        p_dist = heavy_atoms[:, None, :, :] - heavy_atoms[:, :, None, :]
        p_dist = np.linalg.norm(p_dist, axis=-1)

        # select only one pair of pairwise difference
        r, c = np.triu_indices(10, k=1)
        p_dist = p_dist[:, r, c]  # [T, 45]
        p_dist_normalized = p_dist - p_dist.mean(axis=0, keepdims=True)
        p_dist_standardized = p_dist_normalized / p_dist.std(axis=0, keepdims=True)

        kur = scipy.stats.kurtosis(p_dist, axis=0)

        p_dist = p_dist_standardized[:, (kur < self.gmvae_kurtosis_threshold)]

        obs_data = np.concatenate([ram_traj, p_dist], axis=-1)

        return obs_data, ram_traj

    def pwise_dist(self, raw_data):
        raw_data = np.array(raw_data)

        # select heavy atoms
        selected_atoms = raw_data[:, self.selected_atoms, :]

        # Pairwise difference
        p_dist = selected_atoms[:, None, :, :] - selected_atoms[:, :, None, :]
        p_dist = np.linalg.norm(p_dist, axis=-1)

        # select only one pair of pairwise difference
        r, c = np.triu_indices(len(self.selected_atoms), k=1)
        p_dist = p_dist[:, r, c]  # [*, _]
        p_dist_normalized = p_dist - p_dist.mean(axis=0, keepdims=True)
        p_dist_standardized = p_dist_normalized / p_dist.std(axis=0, keepdims=True)

        return p_dist_standardized  # [*, _]

    def compute_dihedral_angle(self, x, indices):
        p0, p1, p2, p3 = (x[:, i] for i in indices)
        b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2

        b1b2 = np.cross(b1, b2)
        b2b3 = np.cross(b2, b3)
        b1b2b3 = np.cross(b1b2, b2b3)

        b2_ = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)

        # dot products
        d = (b1b2b3 * b2_).sum(axis=-1, keepdims=True)
        e = (b1b2 * b2b3).sum(axis=-1, keepdims=True)

        psi = np.arctan2(d, e)

        return np.degrees(psi)


class IonChannel:
    """
    Preprocesses Ion Channel Data data
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        gen_params = kwargs["generation"]

        self.T = gen_params.get("T", 100)  # Number of observations in each timeseries
        self.generation_dt = gen_params.get(
            "generation_dt", 0.0002
        )  # dt data was originally generated with
        self.burn_in = gen_params.get("burn_in", 5000)
        self.small_data_range = gen_params.get("small_data_range", [9200, 14200])
        self.raw_data_path = gen_params.get("raw_data_path")

        # test, validation, prediction test set sizes for complete trajectory data
        self.B_comp_traj_test = gen_params.get("B_comp_traj_test", 1)
        self.B_comp_traj_val = gen_params.get("B_comp_traj_val", 1)
        self.B_comp_traj_pred_test = gen_params.get("B_comp_traj_pred_test", 1)

        # test, validation, prediction test set sizes for 1s trajectory data
        self.B_small_traj_pred_test = gen_params.get("B_small_traj_pred_test", 1)
        self.B_small_traj_test = gen_params.get("B_small_traj_test", 1)
        self.B_small_traj_val = gen_params.get("B_small_traj_val", 1)

        # Path for saving generated data
        use_default_dir = gen_params.get("use_default_dir", True)
        if use_default_dir is True:
            self.data_path = os.path.join(data_path, kwargs["name"])
        else:
            self.data_path = os.path.join(
                os.getcwd(), gen_params["non_default_dir"], kwargs["name"]
            )

    def generate_data(self):
        raw_data = torch.from_numpy(np.loadtxt(self.raw_data_path))  # [*, ]
        raw_data = np.reshape(raw_data, (-1, 1))

        # complete trajectory, create training, test, validation and prediction test set
        comp_train, comp_test, comp_val, comp_pred_test = self.comp_traj_data_split(raw_data)

        # 1s trajectory from Koehs et al.
        small_train, small_test, small_val, small_pred_test = self.small_traj_data_split(raw_data)

        # Save data
        data_description = [
            "comp_train",
            "comp_test",
            "comp_val",
            "comp_pred_test",
            "small_train",
            "small_test",
            "small_val",
            "small_pred_test",
        ]
        data = [
            comp_train,
            comp_test,
            comp_val,
            comp_pred_test,
            small_train,
            small_test,
            small_val,
            small_pred_test,
        ]

        save_data(data, data_description, self.data_path, self.params)

    def small_traj_data_split(self, data):
        """
        data: [*, D]
        return:
            training set: [small_data_range/T, T, 1+1],
            test set: [B_small_traj_test, T, 1+1],
            validation set: [B_small_traj_val, T, 1+1],
            prediction test set: [B_small_traj_pred_test, 2*T, 1+1]
        """
        traj_len, D = data.shape

        # train on 5000 obs length trajectory of Koehs et al.
        train_data = data[self.small_data_range[0] : self.small_data_range[1]]
        train = train_data.reshape(-1, self.T, D)

        # Append observation times
        obs_times_data = np.arange(start=0, stop=self.T, step=1) * self.generation_dt
        obs_times_data = np.broadcast_to(obs_times_data[None, :, None], (train.shape[0], self.T, 1))
        train = np.concatenate([obs_times_data, train], axis=-1)

        # create test, validation and prediction test set out of rest of trajectory
        data = np.delete(data, np.arange(start=0, stop=14200), axis=0)

        # drop leftover data after batching
        l = int(data.shape[0] / self.T) * self.T
        data = data[:l]

        # create batches
        data = data.reshape(-1, self.T, D)

        # split off test series of twice the length, making sure they don't overlap
        l = int(data.shape[0] / 2)
        rand_ind = (
            np.random.choice(
                np.arange(l, step=1), size=self.B_small_traj_pred_test, replace=False
            ).astype(int)
            * 2
        )  # first half of prediction indices
        rand_ind_ = rand_ind + 1

        pred_test, pred_test_ = (
            data[rand_ind],
            data[rand_ind_],
        )  # [B_small_traj_pred_test, T, D], [B_small_traj_pred_test, T, D]
        pred_test = np.concatenate(
            [pred_test, pred_test_], axis=-2
        )  # [B_small_traj_pred_test, 2*T, D]

        # delete test data from data
        data = np.delete(data, np.concatenate([rand_ind, rand_ind_]), axis=0)

        # Append observation times
        obs_times_data = np.arange(start=0, stop=self.T, step=1) * self.generation_dt
        obs_times_data = np.broadcast_to(obs_times_data[None, :, None], (data.shape[0], self.T, 1))
        obs_times_pred_test = np.arange(start=0, stop=2 * self.T, step=1) * self.generation_dt
        obs_times_pred_test = np.broadcast_to(
            obs_times_pred_test[None, :, None], (pred_test.shape[0], 2 * self.T, 1)
        )

        data = np.concatenate([obs_times_data, data], axis=-1)
        pred_test_data = np.concatenate([obs_times_pred_test, pred_test], axis=-1)

        # test, validation split
        test, val = (
            data[: self.B_small_traj_test],
            data[self.B_small_traj_test : self.B_small_traj_test + self.B_small_traj_val],
        )

        return (
            train.astype(float),
            test.astype(float),
            val.astype(float),
            pred_test_data.astype(float),
        )

    def comp_traj_data_split(self, data):
        """
        data: [*, D]
        return:
            training set: [B_comp_traj_train, T, 1+1],
            test set: [B_comp_traj_test, T, 1+1],
            validation set: [B_comp_traj_val, T, 1+1],
            prediction test set: [B_comp_traj_pred_test, 2*T, 1+1]
        """
        traj_len, D = data.shape

        # remove burn-in period
        data = data[self.burn_in :]

        # drop leftover data after batching
        l = int(data.shape[0] / self.T) * self.T
        data = data[:l]

        # create batches
        data = data.reshape(-1, self.T, D)

        # split off test series of twice the length, making sure they don't overlap
        l = int(data.shape[0] / 2)
        rand_ind = (
            np.random.choice(
                np.arange(l, step=1), size=self.B_small_traj_pred_test, replace=False
            ).astype(int)
            * 2
        )  # first half of prediction indices
        rand_ind_ = rand_ind + 1

        pred_test, pred_test_ = (
            data[rand_ind],
            data[rand_ind_],
        )  # [B_comp_traj_pred_test, T, D], [B_comp_traj_pred_test, T, D]
        pred_test = np.concatenate(
            [pred_test, pred_test_], axis=-2
        )  # [B_comp_traj_pred_test, 2*T, D]

        # delete test data from data
        data = np.delete(data, np.concatenate([rand_ind, rand_ind_]), axis=0)

        # Append observation times
        obs_times_data = np.arange(start=0, stop=self.T, step=1) * self.generation_dt
        obs_times_data = np.broadcast_to(obs_times_data[None, :, None], (data.shape[0], self.T, 1))
        obs_times_pred_test = np.arange(start=0, stop=2 * self.T, step=1) * self.generation_dt
        obs_times_pred_test = np.broadcast_to(
            obs_times_pred_test[None, :, None], (pred_test.shape[0], 2 * self.T, 1)
        )

        data = np.concatenate([obs_times_data, data], axis=-1)
        pred_test_data = np.concatenate([obs_times_pred_test, pred_test], axis=-1)

        # train, test, validation split
        B_comp_traj_train = data.shape[0] - self.B_comp_traj_test - self.B_comp_traj_val
        splits = [B_comp_traj_train, B_comp_traj_train + self.B_comp_traj_test]
        train, test, val = np.split(data, splits)

        return (
            train.astype(float),
            test.astype(float),
            val.astype(float),
            pred_test_data.astype(float),
        )


if __name__ == "__main__":
    main()

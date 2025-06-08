import numpy as np
import torch
import os

import warnings
from typing import Union, Optional
import pickle


class FlexiBatch:
    def __init__(
        self,
        action_mask: Optional[list] = None,
        action_mask_: Optional[list] = None,
        terminated: Union[np.ndarray, None] = None,
        memory_weights: Union[np.ndarray, None] = None,
        verbose: bool = True,
        registered_vals: dict = {},
    ):
        self._registered_vals = {}
        self.action_mask = action_mask
        self.action_mask_ = action_mask_
        self.terminated = terminated
        self.memory_weights = memory_weights
        self.verbose = verbose
        self._floats = []
        for rf in registered_vals.keys():
            self._floats.append(rf)
            self._registered_vals[rf] = registered_vals[rf]

    def __getattr__(self, item):
        if item in self._registered_vals:
            return self._registered_vals[item]
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def to_torch(self, device):
        for f in self._floats:
            self._registered_vals[f] = (
                None
                if self._registered_vals[f] is None
                else torch.from_numpy(self._registered_vals[f]).to(device)
            )
        if self.terminated is not None:
            self.terminated = torch.from_numpy(self.terminated).float().to(device)
        if self.memory_weights is not None:
            self.memory_weights = (
                torch.from_numpy(self.memory_weights).float().to(device)
            )

        if self.action_mask is not None:
            assert (
                self.action_mask_ is not None
            ), "If action_mask is not None, then action_mask_ must also be set"
            for i, a in enumerate(self.action_mask):
                self.action_mask[i] = torch.from_numpy(a).float().to(device)
            for i, a in enumerate(self.action_mask_):
                self.action_mask_[i] = torch.from_numpy(a).float().to(device)

    def __str__(self):
        s = ""
        s += f"action_mask: {self.action_mask is not None}\n"
        s += f"action_mask_: {self.action_mask_ is not None}\n"
        for f in self._floats:
            s += f"{f}: {self._registered_vals[f] is not None}\n"
        s += f"terminated: {self.terminated is not None}\n"
        s += f"memory_weights: {self.memory_weights is not None}\n"

        if not self.verbose:
            s += " To see preview of the contents of each array set 'verbose' to True"

        if self.verbose:
            s += f"action_mask: {self.action_mask}\n"
            s += f"action_mask_: {self.action_mask_}\n"
            for f in self._floats:
                s += f"{f}: {self._registered_vals[f]}\n"
        return s


class FlexibleBuffer:
    """Flexible Buffer supports numpy and torch tensor outputs formats,
    but all memories are held internally as numpy buffers because
    Torch.from_numpy() will share the same memory either way in RAM

    Flexible Buffer stores the memories of multiple 'n_agents' agents
    in their own separate memory blocks where each agent has 'num_steps'
    storage capacity. Setting 'n_agents' to 1 will remove a dimension
    from the returned buffer results for single agent tasks.

    cardinal supports both continuous and discrete actions at the
    same time along and it can sample episodes for use in recurrent training
    or policy gradient methods using recorded discounted episodic rewards,
    'G'. cardinal can also store action masks for environments with
    illegal actions and a second reward signal called 'global_auxiliary_reward' for
    simultaneous human and MDP rewards for RLHF + RL.

    For Mixed discrete and continuous actions, actions will be saved and
    returned in the format
        discrete_actions
            [   # Discrete action tensor
                [d0_s0,d1_s0,d2_s0,...,dN-1_s0],
                [d0_s1,d1_s1,d2_s1,...,dN-1_s1],
                            ...,
                [d0_sB,d1_sB,d2_sB,...,dN-1_sB],
            ],
        continuous_actions
            [   # Continuous Action Tensor
                [c0_s0,c1_s0,c2_s0,...,cM-1_s0],
                [c0_s1,c1_s1,c2_s1,...,cM-1_s1],
                            ...,
                [c0_sB,c1_sB,c2_sB,...,cM-1_sB]
            ],

    where d0_s0 refers to discrete dimension 0 out of 'N' dimensions
    sample 0 out of 'B' batch size timesteps. c2_s1 would refer to continuous
    dimension 2 our of 'M' sample timestep 1 our of 'B' batch size.

    init variables:
        num_steps: int Number of timesteps per agent to be saved in the
            buffer.
        obs_size: int Number of dimensions in the flattened 1 dimensional
            observation for a particular agent
        global_auxiliary_reward=False: bool Whether to record a second reward signal for
            human feedback.
        action_mask: [bool] List for whether to mask each dimension of the
            discrete portion of the actions.
        discrete_action_cardinalities: [int] List of integers to denote the
            number of discrete action choices for each discrete action output
        continuous_action_dimension: int Number of continuous action dimensions
            (Note: suppose a network outputs a distribution for each
            continuous dimension like [mean,std], then the continuous_action_dimension
            should be set to 2*n_action_dimensions because flexibuff will save
            exactly as many numbers as specified here)
        path: String the path to where flexibuff will be saved if a path is not
            passed at save time if no such path exists it will be made. default
            is './default_dir/'
        name: the name which will be appended onto the path to save these numpy
            arrays. default is 'flexibuff_test'
        n_agents: int The number of agents to save buffers for.
        state_size: int The number of dimensions for a global state for use in
            centralized training. None by default assuming observations are local
        global_reward: bool reward given to a group of agents
        global_auxiliary_reward: bool a second global reward such as human feedback
        individual_reward: bool reward given to an individual agent
        individual_auxiliary_reward: bool second reward given to individual such as
            with human feedback
        log_prob_discrete: bool whether to track log probabilities for discrete action
            space
        log_prob_continuous: int = 0 the dimension of probabilities to track for
            continuous action spaces for instance if there is one continuous action
            parameterized by a normal distribution with mean mu and std sigma, then
            continuous_action_dimension = 2, but log_prob_continuous would only be
            storing a single probability so it would be 1.
        memory_weights: bool whether or not to store weights along with each timestep
            for memory weighting or another purpose
    """

    def __init__(
        self,
        num_steps: int = 10,
        n_agents=5,
        discrete_action_cardinalities=None,
        track_action_mask=False,  # True if discrete action masks are to be recorded
        path: str = "./default_dir/",
        name: str = "flexibuff_test",
        memory_weights: bool = False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
            "state": ([59], np.float32),
            "state_": ([59], np.float32),
        },
        individual_registered_vars={
            "ind_value_estimates": (None, np.float32),
            "individual_rewards": (None, np.float32),
            "obs": ([59], np.float32),
            "obs_": ([59], np.float32),
            "discrete_log_probs": (None, np.float32),
            "continuous_log_probs": (None, np.float32),
            "discrete_actions": ([3], np.int32),
            "continuous_actions": ([2], np.float32),
        },
    ):
        self._registered_vals = {}
        if track_action_mask:
            assert (
                discrete_action_cardinalities is not None
            ), "If track_action_mask is True, then discrete_action_cardinalities must be set"
        self.num_agents = n_agents
        self.num_steps = num_steps
        self.path = path
        self.name = name
        self.discrete_action_cardinalities = discrete_action_cardinalities
        self.track_action_mask = track_action_mask
        self.mem_size = num_steps

        # For memory weighting
        self.memory_weights = None
        if memory_weights:
            self.memory_weights = np.ones(num_steps, dtype=np.float32)
        self.irvs = []
        self.grvs = []
        # "terminated","memory_weights",

        for grv_key in global_registered_vars.keys():
            self.grvs.append(grv_key)
            shape = [num_steps]
            if global_registered_vars[grv_key][0] is not None:
                shape = shape + global_registered_vars[grv_key][0]
            self._registered_vals[grv_key] = np.zeros(
                shape=shape, dtype=global_registered_vars[grv_key][1]
            )

        for irv_key in individual_registered_vars.keys():
            self.irvs.append(irv_key)
            shape = [n_agents, num_steps]
            if individual_registered_vars[irv_key][0] is not None:
                shape = shape + individual_registered_vars[irv_key][0]
            self._registered_vals[irv_key] = np.zeros(
                shape=shape, dtype=individual_registered_vars[irv_key][1]
            )

        # Create action masks
        self.action_mask = None
        self.action_mask_ = None
        if self.track_action_mask:
            self.action_mask = []
            self.action_mask_ = []
            assert (
                self.discrete_action_cardinalities is not None
            ), "If track_action_mask is True, then discrete_action_cardinalities must be set"

            for dac in self.discrete_action_cardinalities:
                self.action_mask.append(
                    np.ones((n_agents, num_steps, dac), dtype=np.float32)
                )
                self.action_mask_.append(
                    np.ones((n_agents, num_steps, dac), dtype=np.float32)
                )

        # Indicates terminal steps of the environment not truncation
        self.terminated = np.zeros((num_steps), dtype=np.float32)

        # The current save index of where new experience will be added to the
        # buffer
        self.idx = 0
        # The largest recorded save index of the buffer which will be the size
        # of the buffer - 1 once the buffer loops back around to the beginning
        self.steps_recorded = 0

        self.episode_inds = None  # Track for efficient sampling later
        self.episode_lens = None

    def __getattr__(self, item):
        if item in self._registered_vals:
            return self._registered_vals[item]
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def _between(self, num, lastidx, idx):
        lower = lastidx
        upper = idx
        if idx < lastidx:
            upper = lastidx + idx + self.mem_size - lastidx
            if num < idx:
                return True
        if num < upper and num >= lower:
            return True
        return False

    def _update_episode_index(self, idx):
        if idx < self.steps_recorded:
            self.steps_recorded = self.mem_size
        else:
            self.steps_recorded = idx
        # If nothing has been recorded yet, initialize things
        if self.episode_lens is None and self.episode_inds is None:
            self.episode_inds: list[int] | None = [0]
            self.episode_lens: list[int] | None = []
            self.episode_lens.append(idx)
            self.episode_inds.append(idx)
        else:
            assert (self.episode_inds is not None) and (
                self.episode_lens is not None
            ), "Episode indices or lengs are None, make sure to initialize them before saving transitions"
            while self._between(self.episode_inds[0], self.episode_inds[-1], idx):
                self.episode_inds.pop(0)
                self.episode_lens.pop(0)
            if idx < self.episode_inds[-1]:
                self.episode_lens.append(idx + self.mem_size - self.episode_inds[-1])
            else:
                self.episode_lens.append(idx - self.episode_inds[-1])
            self.episode_inds.append(idx)

    def save_transition(
        self,
        terminated: Union[bool, int, float] = False,
        action_mask=None,
        action_mask_=None,
        memory_weight=1.0,
        registered_vals={
            "global_rewards": 1.0,
            "state": np.ones(59),
            "state_": np.ones(59) + 1,
            "ind_value_estimates": np.ones(5),
            "individual_rewards": np.ones(5) + 1,
            "obs": np.ones((5, 59)) - 2,
            "obs_": np.ones((5, 59)) - 3,
            "discrete_log_probs": -np.ones(5),
            "continuous_log_probs": np.ones(5) - 3,
            "discrete_actions": np.zeros((5, 3), dtype=np.int32),
            "continuous_actions": np.zeros((5, 2)) + 0.5,
        },
    ):
        """
        Saves a step into the multi-agent memory buffer

        For inputs [obs,
                    obs_,
                    discrete_actions,
                    continuous_actions,
                    individual_rewards,
                    individual_auxiliary_rewards,
                    ]
        The first dimension 'n_agents' can be a list or numpy array


        obs: [[float,]] The observations for each agent, should have shape [n_agents, obs_size]
        discrete_actions: [[int32,]] The discrete portion of actions taken by the agents
            with shape [n_agents, len(discrete_action_cardinalities)]
        continuous_actions: [[float32]] The continuous portion of the actions taken by the
            agents with shape [n_agents, continuous_action_dimension]
        global_reward: float the reward obtained by the group at this timestep
        global_auxiliary_reward: float A second reward signal for rlhf or other similar uses
        individual_rewards: [float] The rewards obtained by each agent at this timestep
        indivdual_auxiliary_rewards: [float] The second reward signal for each agent at this
            timestep
        action_mask: [[float]] The list of legal actions for each agent at obs with shape
            [len(discrete_action_cardinality), n_agents, discrete_action_cardinality[i]]
        action_mask_: The same for next state
        state: [float] The global state for centralized training methods with shape [state_size]
        state_: [float] The same as state but the next step
        """

        self.idx = self.idx % self.mem_size

        for k in self.irvs:
            if k not in registered_vals:
                warnings.warn(
                    f"Warning, individual registered value '{k}' not present in registered values so it is not being updated"
                )
        for k in self.grvs:
            if k not in registered_vals:
                warnings.warn(
                    f"Warning, global registered value '{k}' not present in registered values so it is not being updated"
                )

        for k in registered_vals.keys():
            if k in self.irvs:
                # print(
                #    f"param: {k} in irv: shape{self._registered_vals[k][:, self.idx].shape}, registered vals shape: {registered_vals[k].shape}"
                # )
                self._registered_vals[k][:, self.idx] = registered_vals[k]
            elif k in self.grvs:
                # print(
                #     f"param: {k} in grv: shape{self._registered_vals[k][self.idx].shape}, registered vals shape: {registered_vals[k].shape}"
                # )
                self._registered_vals[k][self.idx] = registered_vals[k]
            else:
                warnings.warn(
                    f"Warning, passed value '{k}' not present in self individual or global values. Make sure it was registered at init()"
                )

        if self.action_mask is not None and self.action_mask_ is not None:
            assert (
                self.discrete_action_cardinalities is not None
            ), "If action_mask is not None, then discrete_action_cardinalities must be set"
            if action_mask is None or action_mask_ is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.action_mask[{self.idx}] and/or FlexibleBuffer.action_mask_[{self.idx}] to nan because keyword argument FlexibleBuffer.action_mask=None"
                )
                for i in range(len(self.discrete_action_cardinalities)):
                    self.action_mask[i][:, self.idx] = np.nan
                    self.action_mask_[i][:, self.idx] = np.nan
            else:
                for i, dac in enumerate(self.discrete_action_cardinalities):
                    self.action_mask[i][:, self.idx] = action_mask[i]
                    self.action_mask_[i][:, self.idx] = action_mask_[i]

        if self.memory_weights is not None:
            self.memory_weights[self.idx] = memory_weight

        if terminated:
            self._update_episode_index(self.idx + 1)  # For efficient sampling later

        self.terminated[self.idx] = float(terminated)
        self.idx += 1
        self.steps_recorded = max(self.idx, self.steps_recorded)

    def sample_transitions(
        self, batch_size=256, weighted=False, as_torch=False, device="cuda", idx=None
    ):
        size = min(self.steps_recorded, batch_size)
        if idx is None:
            if weighted:
                assert (
                    self.memory_weights is not None
                ), "Memory weights must be set to sample weighted transitions"
                idx = np.random.choice(
                    min(self.steps_recorded, self.mem_size),
                    size,
                    replace=False,
                    p=self.memory_weights[0 : self.steps_recorded],
                )
            else:
                idx = np.random.choice(
                    min(self.steps_recorded, self.mem_size), size, replace=False
                )

        if self.action_mask is not None and self.action_mask_ is not None:
            assert (
                self.discrete_action_cardinalities is not None
            ), "If action_mask is not None, then discrete_action_cardinalities must be set"
            action_mask = []
            action_mask_ = []
            for i, dac in enumerate(self.discrete_action_cardinalities):
                action_mask.append(self.action_mask[i][:, idx])
                action_mask_.append(self.action_mask_[i][:, idx])
        else:
            action_mask_ = None
            action_mask = None

        registered_vals = {}
        for grv in self.grvs:
            registered_vals[grv] = self._registered_vals[grv][idx]
        for irv in self.irvs:
            registered_vals[irv] = self._registered_vals[irv][:, idx]

        fb = FlexiBatch(
            action_mask=action_mask,
            action_mask_=action_mask_,
            terminated=self.terminated[idx] if self.terminated is not None else None,
            memory_weights=(
                self.memory_weights[idx] if self.memory_weights is not None else None
            ),
            registered_vals=registered_vals,
        )

        if as_torch:
            fb.to_torch(device=device)

        return fb

    def sample_episodes(
        self, max_batch_size=256, as_torch=False, device="cuda", n_episodes=None
    ):
        if self.episode_inds is None or self.episode_lens is None:
            warnings.warn(
                "Episode indices and lengths are not set, returning empty list"
            )
            return []
        tempidx = self.episode_inds.copy()
        templen = self.episode_lens.copy()

        batch_idx = []
        batch_len = []
        tot_size = 0
        if n_episodes is None:
            n_episodes = len(templen)
        while (
            tot_size < max_batch_size
            and len(templen) > 0
            and len(batch_idx) < n_episodes
        ):
            i = np.random.randint(0, len(templen))
            batch_idx.append(tempidx.pop(i))
            batch_len.append(min(templen.pop(i), max_batch_size - tot_size))
            tot_size += batch_len[-1]

        episodes = []
        for i in range(len(batch_idx)):
            idx = np.mod(
                np.arange(batch_idx[i], batch_idx[i] + batch_len[i]), self.mem_size
            )
            episodes.append(
                self.sample_transitions(as_torch=as_torch, idx=idx, device=device)
            )
        return episodes

    def print_idx(self, idx):
        print(
            f"obs: {self.obs[idx]} | obs_ {self.obs_[idx]}, action: {self.discrete_actions[idx]}, reward: {self.global_rewards[idx]}, done: {self.terminated[idx]}"
            + f", legal: {self.action_mask[idx]}, legal_: {self.action_mask_[idx]}"
            if self.action_mask is not None and self.action_mask_ is not None
            else ""
        )

    @staticmethod
    def G(rewards: torch.Tensor, terminated: torch.Tensor, last_value=0, gamma=0.99):
        G = torch.zeros_like(rewards).to(rewards.device)
        G[-1] = rewards[-1]
        if terminated[-1] < 0.5:
            G[-1] += gamma * last_value

        for i in range(len(rewards) - 2, -1, -1):
            G[i] = rewards[i] + gamma * G[i + 1] * (1 - terminated[i])
        G = G.unsqueeze(-1)
        return G

    @staticmethod
    def GAE(
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
        last_value=0,
        gamma=0.99,
        gae_lambda=0.95,
    ):

        advantages = torch.zeros_like(rewards).to(rewards.device)
        num_steps = len(rewards)
        last_gae_lam = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_non_terminal = 1.0 - terminated[-1]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - terminated[step]
                next_values = values[step + 1]
            delta = (
                rewards[step] + gamma * next_values * next_non_terminal - values[step]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        G = advantages + values

        return G.unsqueeze(-1), advantages.unsqueeze(-1)

    @staticmethod
    def K_Step_TD(
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminated: torch.Tensor,
        last_value=0,
        gamma=0.99,
        k=1,
    ):
        G = torch.zeros_like(rewards).to(rewards.device)
        n_step = len(rewards)
        for step in range(n_step - 1, -1, -1):
            v = 0
            for i in range(min(k, n_step - step)):
                if i + step == n_step - 1:
                    v += gamma ** (i) * rewards[-1] + (
                        gamma ** (i + 1)
                    ) * last_value * (1 - terminated[-1])

                else:
                    if i == k - 1:
                        v += (gamma ** (i)) * rewards[step + i] + (
                            gamma ** (i + 1)
                        ) * values[step + i + 1] * (1 - terminated[step + i])

                    else:
                        if terminated[step + i]:
                            v += (gamma**i) * rewards[step + i]
                            break
                        else:
                            v += (gamma**i) * rewards[step + i]
            G[step] = v
            # print(-step - 1)
        return G.unsqueeze(-1), (G - values).unsqueeze(-1)

    @staticmethod
    def load(path, name):
        if not os.path.exists(path) or not os.path.exists(path + name + "_idx.npy"):
            print(
                f"path '{path}' or '{path + name + '_idx.npy'}' does not exist yet. returning empty buffer"
            )
            return FlexibleBuffer()
        fb = FlexibleBuffer()
        fb.path = path
        fb.name = name

        fb.idx = np.load(path + name + "_idx.npy")
        fb.steps_recorded = np.load(path + name + "_steps_recorded.npy")
        fb.mem_size = np.load(path + name + "_mem_size.npy")
        fb.track_action_mask = pickle.load(
            open(fb.path + fb.name + "track_action_mask", "rb")
        )
        fb.irvs = pickle.load(open(fb.path + fb.name + "irvs", "rb"))
        fb.grvs = pickle.load(open(fb.path + fb.name + "grvs", "rb"))

        if os.path.exists(path + name + "_discrete_action_cardinalities.npy"):
            fb.discrete_action_cardinalities = np.load(
                fb.path + fb.name + "_discrete_action_cardinalities.npy"
            )
        else:
            fb.discrete_action_cardinalities = None

        for param in fb.irvs + fb.grvs:
            if os.path.exists(fb.path + fb.name + "_" + param + ".npy"):
                fb._registered_vals[param] = np.load(
                    fb.path + fb.name + "_" + param + ".npy"
                )
            else:
                print(fb.path + fb.name + "_" + param + ".npy was not found")
                fb._registered_vals[param] = None
        if os.path.exists(fb.path + fb.name + "_terminated.npy"):
            fb.terminated = np.load(fb.path + fb.name + "_terminated.npy")
        else:
            fb.terminated = None
        if os.path.exists(fb.path + fb.name + "_memory_weights.npy"):
            fb.memory_weights = np.load(fb.path + fb.name + "_memory_weights.npy")
        else:
            fb.memory_weights = None

        fb.action_mask = []
        fb.action_mask_ = []
        if fb.track_action_mask and fb.discrete_action_cardinalities is not None:
            for i, dac in enumerate(fb.discrete_action_cardinalities):
                if os.path.exists(
                    fb.path + fb.name + f"_action_mask{i}.npy"
                ) and os.path.exists(fb.path + fb.name + f"_action_mask_{i}.npy"):
                    fb.action_mask.append(
                        np.load(fb.path + fb.name + f"_action_mask{i}.npy")
                    )
                    fb.action_mask_.append(
                        np.load(fb.path + fb.name + f"_action_mask_{i}.npy")
                    )
                else:
                    print(fb.path + fb.name + f"_action_mask{i}.npy not found")
                    fb.action_mask = None
                    fb.action_mask_ = None
                    break

        fb.episode_inds = np.load(fb.path + fb.name + "_episode_inds.npy").tolist()
        fb.episode_lens = np.load(fb.path + fb.name + "_episode_lens.npy").tolist()

        return fb

    @staticmethod
    def save(fb):
        if not os.path.exists(fb.path):
            print(f"Path did not exist so making '{fb.path}'")
            os.makedirs(fb.path)

        np.save(fb.path + fb.name + "_idx.npy", np.array(fb.idx))
        np.save(fb.path + fb.name + "_steps_recorded.npy", np.array(fb.steps_recorded))
        np.save(fb.path + fb.name + "_mem_size.npy", np.array(fb.mem_size))
        np.save(fb.path + fb.name + "_episode_inds.npy", np.array(fb.episode_inds))
        np.save(fb.path + fb.name + "_episode_lens.npy", np.array(fb.episode_lens))
        pickle.dump(fb.irvs, open(fb.path + fb.name + "irvs", "wb"))
        pickle.dump(fb.grvs, open(fb.path + fb.name + "grvs", "wb"))
        pickle.dump(
            fb.track_action_mask, open(fb.path + fb.name + "track_action_mask", "wb")
        )

        for param in fb.irvs + fb.grvs:
            if fb._registered_vals[param] is not None:
                np.save(
                    fb.path + fb.name + "_" + param + ".npy", fb._registered_vals[param]
                )

        if fb.terminated is not None:
            np.save(fb.path + fb.name + "_terminated.npy", fb.terminated)
        if fb.memory_weights is not None:
            np.save(fb.path + fb.name + "_memory_weights.npy", fb.memory_weights)

        if fb.track_action_mask:
            for i, dac in enumerate(fb.discrete_action_cardinalities):
                if fb.action_mask is not None:
                    np.save(
                        fb.path + fb.name + f"_action_mask{i}.npy", fb.action_mask[i]
                    )
                    np.save(
                        fb.path + fb.name + f"_action_mask_{i}.npy",
                        fb.action_mask_[i],
                    )

        if fb.discrete_action_cardinalities is not None:
            np.save(
                fb.path + fb.name + "_discrete_action_cardinalities.npy",
                np.array(fb.discrete_action_cardinalities),
            )

    @staticmethod
    def sum_batch_rewards(batch: FlexiBatch, agent_num: int = 0):
        bgr = 0 if batch.global_rewards is None else batch.global_rewards
        bgar = (
            0
            if batch.global_auxiliary_rewards is None
            else batch.global_auxiliary_rewards
        )
        bir = (
            0
            if batch.individual_rewards is None
            else batch.individual_rewards[agent_num]
        )
        biar = (
            0
            if batch.individual_auxiliary_rewards is None
            else batch.individual_auxiliary_rewards[agent_num]
        )
        return bgr + bgar + bir + biar

    def __str__(self):
        s = f"Buffer size: {self.mem_size}, steps_recorded: {self.steps_recorded}, {self.steps_recorded/self.mem_size*100}%, current idx: {self.idx} \n"
        s += f"discrete_action_cardinalities: {self.discrete_action_cardinalities}\n"
        s += f"action_mask: {self.action_mask is not None}\n"
        s += f"action_mask_: {self.action_mask_ is not None}\n"
        s += f"terminated: {self.terminated is not None}\n"
        s += f"memory_weights: {self.memory_weights is not None}\n"
        for param in self.irvs + self.grvs:
            s += f"{param}: {self._registered_vals[param] is not None}\n"

        s += f"action_mask: {self.action_mask}\n"
        s += f"action_mask_: {self.action_mask_}\n"
        s += f"terminated: {self.terminated}\n"
        s += f"memory_weights: {self.memory_weights}\n"
        for param in self.irvs + self.grvs:
            s += f"{param}: {self._registered_vals[param]}\n"
        return s

    def reset(self):
        self.idx = 0
        self.steps_recorded = 0
        self.episode_inds = None
        self.episode_lens = None

    def summarize_buffer(self):
        print("Depricated, doesn't work anymore")
        return 0
        # n_elem = (
        #     np.size(self.discrete_actions)
        #     + np.size(self.obs) * 2
        #     + np.size(self.global_rewards)
        #     + np.size(self.terminated)
        # )
        # if self.action_mask is not None:
        #     n_elem += np.size(self.action_mask) * 2
        # er = ""
        # if self.global_auxiliary_rewards is not None:
        #     er = "*2"
        #     n_elem += np.size(self.global_auxiliary_rewards)
        # print(
        #     f"Buffer size: {self.steps_recorded} / {self.mem_size} steps. Current idx: {self.idx}. discrete_action_cardinalities: {self.discrete_action_cardinalities}, state: {self.obs.shape[1]}"
        # )
        # print(
        #     f"Total elements: actions {np.size(self.discrete_actions)} + states {np.size(self.obs)}*2 + rewards {np.size(self.global_rewards)}{er} + legality {np.size(self.action_mask)}*2 + done {np.size(self.terminated)} = {n_elem}"
        # )

        # if self.steps_recorded == 0:
        #     return

        # start = self.idx % self.steps_recorded
        # rs = []
        # ex_rs = []
        # rs.append(0)
        # ex_rs.append(0)
        # looping = True
        # while looping:
        #     if start == self.idx - 1:
        #         looping = False
        #     rs[-1] += self.global_rewards[start]
        #     if self.global_auxiliary_rewards:
        #         ex_rs[-1] += self.global_auxiliary_rewards[start]

        #     if self.terminated[start] == 1:
        #         rs.append(0)
        #         if self.global_auxiliary_rewards:
        #             ex_rs.append(0)

        #     start += 1
        #     start = start % self.steps_recorded
        #     # print(start)
        #     # print(self.steps_recorded)

        # print(f"Episode Cumulative Rewards: {rs},\nExpert Rewards: {ex_rs}")

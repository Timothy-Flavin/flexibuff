import numpy as np
import torch
import os

import warnings
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class FlexiBatch:
    obs: Union[np.ndarray, torch.FloatTensor, None] = None
    obs_: Union[np.ndarray, torch.FloatTensor, None] = None
    state: Union[np.ndarray, torch.FloatTensor, None] = None
    state_: Union[np.ndarray, torch.FloatTensor, None] = None
    global_rewards: Union[np.ndarray, torch.FloatTensor, None] = None
    global_auxiliary_rewards: Union[np.ndarray, torch.FloatTensor, None] = None
    individual_rewards: Union[np.ndarray, torch.FloatTensor, None] = None
    individual_auxiliary_rewards: Union[np.ndarray, torch.FloatTensor, None] = None
    discrete_actions: Union[np.ndarray, torch.IntTensor, None] = None
    continuous_actions: Union[np.ndarray, torch.FloatTensor, None] = None
    discrete_log_probs: Union[np.ndarray, torch.FloatTensor, None] = None
    continuous_log_probs: Union[np.ndarray, torch.FloatTensor, None] = None
    action_mask: Optional[list] = None
    action_mask_: Optional[list] = None
    terminated: Union[np.ndarray, torch.FloatTensor, None] = None
    memory_weights: Union[np.ndarray, torch.FloatTensor, None] = None
    verbose: bool = True

    _floats = [
        "obs",
        "obs_",
        "state",
        "state_",
        "global_rewards",
        "global_auxiliary_rewards",
        "individual_rewards",
        "individual_auxiliary_rewards",
        "continuous_actions",
        "discrete_log_probs",
        "continuous_log_probs",
        "terminated",
        "memory_weights",
    ]

    def to_torch(self, device):
        for f in self._floats:
            self.__dict__[f] = (
                None
                if self.__dict__[f] is None
                else torch.from_numpy(self.__dict__[f]).float().to(device)
            )
        self.discrete_actions = (
            None
            if self.discrete_actions is None
            else torch.from_numpy(self.discrete_actions).long().to(device)
        )
        if self.action_mask is not None:
            for i, a in enumerate(self.action_mask):
                self.action_mask[i] = torch.from_numpy(a).float().to(device)
            for i, a in enumerate(self.action_mask_):
                self.action_mask_[i] = torch.from_numpy(a).float().to(device)

    def __str__(self):
        s = ""
        s += f"obs: {self.obs is not None}\n"
        s += f"obs_: {self.obs_ is not None}\n"
        s += f"state: {self.state is not None}\n"
        s += f"state_: {self.state_ is not None}\n"
        s += f"global_rewards: {self.global_rewards is not None}\n"
        s += f"global_auxiliary_rewards: {self.global_auxiliary_rewards is not None}\n"
        s += f"individual_rewards: {self.individual_rewards is not None}\n"
        s += f"individual_auxiliary_rewards: {self.individual_auxiliary_rewards is not None}\n"
        s += f"discrete_actions: {self.discrete_actions is not None}\n"
        s += f"continuous_actions: {self.continuous_actions is not None}\n"
        s += f"discrete_log_probs: {self.discrete_log_probs is not None}\n"
        s += f"continuous_log_probs: {self.continuous_log_probs is not None}\n"
        s += f"action_mask: {self.action_mask is not None}\n"
        s += f"action_mask_: {self.action_mask_ is not None}\n"
        s += f"terminated: {self.terminated is not None}\n"
        s += f"memory_weights: {self.memory_weights is not None}\n"

        if not self.verbose:
            s += " To see preview of the contents of each array set 'verbose' to True"

        if self.verbose:
            s += f"obs: {self.obs}\n"
            s += f"obs_: {self.obs_}\n"
            s += f"state: {self.state}\n"
            s += f"state_: {self.state_}\n"
            s += f"global_rewards: {self.global_rewards}\n"
            s += f"global_auxiliary_rewards: {self.global_auxiliary_rewards}\n"
            s += f"individual_rewards: {self.individual_rewards}\n"
            s += f"individual_auxiliary_rewards: {self.individual_auxiliary_rewards}\n"
            s += f"discrete_actions: {self.discrete_actions}\n"
            s += f"continuous_actions: {self.continuous_actions}\n"
            s += f"discrete_log_probs: {self.discrete_log_probs}\n"
            s += f"continuous_log_probs: {self.continuous_log_probs}\n"
            s += f"action_mask: {self.action_mask}\n"
            s += f"action_mask_: {self.action_mask_}\n"
            s += f"terminated: {self.terminated}\n"
            s += f"memory_weights: {self.memory_weights}\n"
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
        continious_action_dimension: int Number of continuous action dimensions
            (Note: suppose a network outputs a distribution for each
            continuous dimension like [mean,std], then the continious_action_dimension
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
        obs_size: int = 1,
        action_mask=False,
        discrete_action_cardinalities=None,
        continious_action_dimension=None,
        path: str = "./default_dir/",
        name: str = "flexibuff_test",
        n_agents: int = 1,
        state_size: int = None,
        global_reward: bool = False,
        global_auxiliary_reward: bool = False,
        individual_reward: bool = False,
        individual_auxiliary_reward: bool = False,
        log_prob_discrete: bool = False,  #
        log_prob_continuous: int = 0,  # int
        memory_weights: bool = False,
    ):
        # assert (
        #    discrete_action_cardinalities is not None
        #    or continious_action_dimension is not None
        # ), "'discrete_action_cardinalities' and 'continious_action_dimension' \
        #    must not both be None so actions may be saved"

        self.num_agents = n_agents
        self.num_steps = num_steps
        self.path = path
        self.name = name
        self.discrete_action_cardinalities = discrete_action_cardinalities
        self.continious_action_dimension = continious_action_dimension
        self.mem_size = num_steps
        self.obs = np.zeros((n_agents, num_steps, obs_size), dtype=np.float32)
        self.obs_ = np.zeros((n_agents, num_steps, obs_size), dtype=np.float32)

        # For memory weighting
        self.memory_weights = None
        if memory_weights:
            self.memory_weights = np.ones(num_steps, dtype=np.float32)

        # State for CTDE
        self.state_size = state_size
        self.state = None
        self.state_ = None
        if state_size is not None:
            self.state = np.zeros((num_steps, state_size), dtype=np.float32)
            self.state_ = np.zeros((num_steps, state_size), dtype=np.float32)

        # If we have discrete actions set up discrete action buffer
        self.discrete_actions = None
        if discrete_action_cardinalities is not None:
            self.discrete_actions = np.zeros(
                (n_agents, num_steps, len(discrete_action_cardinalities)),
                dtype=np.int32,
            )
        self.discrete_log_probs = None
        if log_prob_discrete:
            self.discrete_log_probs = -1.0 * np.ones(
                (n_agents, num_steps, len(discrete_action_cardinalities)),
                dtype=np.float32,
            )

        # If we have continuous actions set up continuous buffer
        self.continuous_actions = None
        if continious_action_dimension is not None:
            self.continuous_actions = np.zeros(
                (n_agents, num_steps, continious_action_dimension),
                dtype=np.float32,
            )
        self.continuous_log_probs = None
        if log_prob_continuous > 0:
            self.continuous_log_probs = -1 * np.ones(
                (n_agents, num_steps, log_prob_continuous),
                dtype=np.float32,
            )

        # Set up reward buffers
        self.global_rewards = None
        self.global_auxiliary_rewards = None
        self.individual_rewards = None
        self.individual_auxiliary_rewards = None

        if global_reward:
            self.global_rewards = np.zeros(num_steps, dtype=np.float32)
        if global_auxiliary_reward:
            self.global_auxiliary_rewards = np.zeros(num_steps, dtype=np.float32)
        if individual_reward:
            self.individual_rewards = np.zeros((n_agents, num_steps), dtype=np.float32)
        if individual_auxiliary_reward:
            self.individual_auxiliary_rewards = np.zeros(
                (n_agents, num_steps), dtype=np.float32
            )

        # Create action masks
        self.action_mask = None
        self.action_mask_ = None
        if action_mask:
            self.action_mask = []
            self.action_mask_ = []
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

        self.array_like_params = [
            "obs",
            "obs_",
            "state",
            "state_",
            "global_rewards",
            "global_auxiliary_rewards",
            "individual_rewards",
            "individual_auxiliary_rewards",
            "discrete_actions",
            "continuous_actions",
            "discrete_log_probs",
            "continuous_log_probs",
            "terminated",
            "memory_weights",
        ]

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
            self.episode_inds = [0]
            self.episode_lens = []
            self.episode_lens.append(idx)
            self.episode_inds.append(idx)
        else:
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
        obs=None,
        obs_=None,
        terminated=None,
        discrete_actions=None,
        continuous_actions=None,
        global_reward=None,
        global_auxiliary_reward=None,
        individual_rewards=None,
        individual_auxiliary_rewards=None,
        action_mask=None,
        action_mask_=None,
        state=None,
        state_=None,
        discrete_log_probs=None,
        continuous_log_probs=None,
        memory_weight=1.0,
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

        if self.obs is not None:
            if obs is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.obs[{self.idx}] to nans because keyword argument obs=None"
                )
            self.obs[:, self.idx] = np.array(obs)
            if obs_ is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.obs_[{self.idx}] to nans because keyword argument obs_=None"
                )
            self.obs_[:, self.idx] = np.array(obs_)

        if self.discrete_actions is not None:
            if discrete_actions is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.discrete_actions[{self.idx}] to nans because keyword argument discrete_actions=None"
                )
            self.discrete_actions[:, self.idx] = np.array(discrete_actions)
        if self.continuous_actions is not None:
            if continuous_actions is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.continuous_actions[{self.idx}] to nans because keyword argument continuous_actions=None"
                )
            self.continuous_actions[:, self.idx] = continuous_actions

        if self.global_rewards is not None:
            if global_reward is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.global_rewards[{self.idx}] to nan because keyword argument global_rewards=None"
                )
            self.global_rewards[self.idx] = global_reward
        if self.global_auxiliary_rewards is not None:
            if global_auxiliary_reward is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.global_auxiliary_rewards[{self.idx}] to nan because keyword argument global_auxiliary_rewards=None"
                )
            self.global_auxiliary_rewards[self.idx] = global_auxiliary_reward

        if self.individual_rewards is not None:
            if individual_rewards is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.individual_rewards[{self.idx}] to nan because keyword argument individual_rewards=None"
                )
            self.individual_rewards[:, self.idx] = np.array(individual_rewards)
        if self.individual_auxiliary_rewards is not None:
            if individual_auxiliary_rewards is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.individual_auxiliary_rewards[{self.idx}] to nan because keyword argument individual_auxiliary_rewards=None"
                )
            self.individual_auxiliary_rewards[:, self.idx] = np.array(
                individual_auxiliary_rewards
            )

        if self.action_mask is not None:
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

        if self.state is not None:
            if state is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.state[{self.idx}] to nan because keyword argument state=None"
                )
            self.state[self.idx] = np.array(state)
            if state_ is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.state_[{self.idx}] to nan because keyword argument state_=None"
                )
            self.state_[self.idx] = np.array(state_)

        if self.discrete_log_probs is not None:
            if discrete_log_probs is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.discrete_log_probs[{self.idx}] to nan because keyword argument discrete_log_probs=None"
                )
                self.discrete_log_probs[:, self.idx] = np.nan
            else:
                self.discrete_log_probs[:, self.idx] = np.array(discrete_log_probs)
        if self.continuous_log_probs is not None:
            if continuous_log_probs is None:
                warnings.warn(
                    f"Warning, setting FlexibleBuffer.continuous_log_probs[{self.idx}] to nan because keyword argument continuous_log_probs=None"
                )
                self.continuous_log_probs[:, self.idx] = np.nan
            else:
                self.continuous_log_probs[:, self.idx] = np.array(continuous_log_probs)

        if self.memory_weights is not None:
            self.memory_weights[self.idx] = memory_weight

        if terminated:
            self._update_episode_index(self.idx + 1)  # For efficient sampling later

        self.terminated[self.idx] = float(terminated)
        self.idx += 1
        self.steps_recorded = max(self.idx, self.steps_recorded)

    def sample_transitions(
        self, batch_size=256, weighted=False, torch=False, device="cuda", idx=None
    ):
        size = min(self.steps_recorded, batch_size)
        if idx is None:
            if weighted:
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

        if self.action_mask is not None:
            action_mask = []
            action_mask_ = []
            for i, dac in enumerate(self.discrete_action_cardinalities):
                action_mask.append(self.action_mask[i][:, idx])
                action_mask_.append(self.action_mask_[i][:, idx])
        else:
            action_mask_ = None
            action_mask = None

        fb = FlexiBatch(
            obs=self.obs[:, idx] if self.obs is not None else None,
            obs_=self.obs_[:, idx] if self.obs_ is not None else None,
            state=self.state[idx] if self.state is not None else None,
            state_=self.state_[idx] if self.state_ is not None else None,
            global_rewards=(
                self.global_rewards[idx] if self.global_rewards is not None else None
            ),
            global_auxiliary_rewards=(
                self.global_auxiliary_rewards[idx]
                if self.global_auxiliary_rewards is not None
                else None
            ),
            individual_rewards=(
                self.individual_rewards[:, idx]
                if self.individual_rewards is not None
                else None
            ),
            individual_auxiliary_rewards=(
                self.individual_auxiliary_rewards[:, idx]
                if self.individual_auxiliary_rewards is not None
                else None
            ),
            discrete_actions=(
                self.discrete_actions[:, idx]
                if self.discrete_actions is not None
                else None
            ),
            continuous_actions=(
                self.continuous_actions[:, idx]
                if self.continuous_actions is not None
                else None
            ),
            discrete_log_probs=(
                self.discrete_log_probs[:, idx]
                if self.discrete_log_probs is not None
                else None
            ),
            continuous_log_probs=(
                self.continuous_log_probs[:, idx]
                if self.continuous_log_probs is not None
                else None
            ),
            action_mask=action_mask,
            action_mask_=action_mask_,
            terminated=self.terminated[idx] if self.terminated is not None else None,
            memory_weights=(
                self.memory_weights[idx] if self.memory_weights is not None else None
            ),
        )

        if torch:
            fb.to_torch(device=device)

        return fb

    def sample_episodes(self, max_batch_size=256, torch=False, device="cuda"):
        tempidx = self.episode_inds.copy()
        templen = self.episode_lens.copy()

        batch_idx = []
        batch_len = []
        tot_size = 0
        while tot_size < max_batch_size and len(templen) > 0:
            i = np.random.randint(0, len(templen))
            batch_idx.append(tempidx.pop(i))
            batch_len.append(templen.pop(i))

        episodes = []
        for i in range(len(batch_idx)):
            idx = np.mod(
                np.arange(batch_idx[i], batch_idx[i] + batch_len[i]), self.mem_size
            )
            episodes.append(self.sample_transitions(torch=torch, idx=idx))
        return episodes

    def print_idx(self, idx):
        print(
            f"obs: {self.obs[idx]} | obs_ {self.obs_[idx]}, action: {self.discrete_actions[idx]}, reward: {self.global_rewards[idx]}, done: {self.terminated[idx]}, legal: {self.action_mask[idx]}, legal_: {self.action_mask_[idx]}"
        )

    # def save_to_drive(self):
    #     if not os.path.exists(self.path):
    #         print(f"Path did not exist so making '{self.path}'")
    #         os.makedirs(self.path)

    #     np.save(self.path + self.name + "_idx.npy", np.array(self.idx))
    #     np.save(
    #         self.path + self.name + "_steps_recorded.npy", np.array(self.steps_recorded)
    #     )
    #     np.save(self.path + self.name + "_mem_size.npy", np.array(self.mem_size))

    #     for param in self.array_like_params:
    #         if self.__dict__[param] is not None:
    #             np.save(
    #                 self.path + self.name + "_" + param + ".npy", self.__dict__[param]
    #             )

    #     for i, dac in enumerate(self.discrete_action_cardinalities):
    #         if self.action_mask is not None:
    #             np.save(
    #                 self.path + self.name + f"_action_mask{i}.npy", self.action_mask[i]
    #             )
    #             np.save(
    #                 self.path + self.name + f"_action_mask_{i}.npy",
    #                 self.action_mask_[i],
    #             )

    #     if self.discrete_actions is not None:
    #         np.save(
    #             self.path + self.name + "_discrete_action_cardinalities.npy",
    #             np.array(self.discrete_action_cardinalities),
    #         )

    # def load_from_drive(self, set_sizes=True):
    #     if not os.path.exists(self.path):
    #         print(f"path '{self.path}' does not exist yet. returning")
    #         return
    #     self.idx = np.load(self.path + self.name + "_idx.npy")
    #     self.steps_recorded = np.load(self.path + self.name + "_steps_recorded.npy")

    #     for param in self.array_like_params:
    #         if os.path.exists(self.path + self.name + "_" + param + ".npy"):
    #             self.__dict__[param] = np.load(
    #                 self.path + self.name + "_" + param + ".npy"
    #             )
    #         else:
    #             print(self.path + self.name + "_" + param + ".npy was not found")
    #             self.__dict__[param] = None

    #     for i, dac in enumerate(self.discrete_action_cardinalities):
    #         if os.path.exists(
    #             self.path + self.name + f"_action_mask{i}.npy"
    #         ) and os.path.exists(self.path + self.name + f"_action_mask_{i}.npy"):
    #             self.action_mask[i] = np.load(
    #                 self.path + self.name + f"_action_mask{i}.npy"
    #             )
    #             self.action_mask_[i] = np.load(
    #                 self.path + self.name + f"_action_mask_{i}.npy"
    #             )
    #         else:
    #             print(self.path + self.name + f"_action_mask{i}.npy not found")
    #             self.action_mask = None
    #             self.action_mask_ = None
    #             break
    #     if os.path.exists(self.path + self.name + "_discrete_action_cardinalities.npy"):
    #         self.discrete_action_cardinalities = np.load(
    #             self.path + self.name + "_discrete_action_cardinalities.npy"
    #         )

    #     self.episode_inds = np.load(
    #         self.path + self.name + "_episode_inds.npy"
    #     ).tolist()
    #     self.episode_lens = np.load(
    #         self.path + self.name + "_episode_lens.npy"
    #     ).tolist()

    @staticmethod
    def load(path, name):
        if not os.path.exists(path) or not os.path.exists(path + name + "_idx.npy"):
            print(
                f"path '{path}' or '{path + name + '_idx.npy'}' does not exist yet. returning"
            )
            return
        fb = FlexibleBuffer()
        fb.path = path
        fb.name = name

        fb.idx = np.load(path + name + "_idx.npy")
        fb.steps_recorded = np.load(path + name + "_steps_recorded.npy")
        fb.mem_size = np.load(path + name + "_mem_size.npy")

        if os.path.exists(path + name + "_discrete_action_cardinalities.npy"):
            fb.discrete_action_cardinalities = np.load(
                fb.path + fb.name + "_discrete_action_cardinalities.npy"
            )

        for param in fb.array_like_params:
            if os.path.exists(fb.path + fb.name + "_" + param + ".npy"):
                fb.__dict__[param] = np.load(fb.path + fb.name + "_" + param + ".npy")
            else:
                print(fb.path + fb.name + "_" + param + ".npy was not found")
                fb.__dict__[param] = None

        fb.action_mask = []
        fb.action_mask_ = []
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

        for param in fb.array_like_params:
            if fb.__dict__[param] is not None:
                np.save(fb.path + fb.name + "_" + param + ".npy", fb.__dict__[param])

        for i, dac in enumerate(fb.discrete_action_cardinalities):
            if fb.action_mask is not None:
                np.save(fb.path + fb.name + f"_action_mask{i}.npy", fb.action_mask[i])
                np.save(
                    fb.path + fb.name + f"_action_mask_{i}.npy",
                    fb.action_mask_[i],
                )

        if fb.discrete_actions is not None:
            np.save(
                fb.path + fb.name + "_discrete_action_cardinalities.npy",
                np.array(fb.discrete_action_cardinalities),
            )

    def __str__(self):
        s = f"Buffer size: {self.mem_size}, steps_recorded: {self.steps_recorded}, {self.steps_recorded/self.mem_size*100}%, current idx: {self.idx} \n"
        s += f"obs: {self.obs is not None}\n"
        s += f"obs_: {self.obs_ is not None}\n"
        s += f"state: {self.state is not None}\n"
        s += f"state_: {self.state_ is not None}\n"
        s += f"global_rewards: {self.global_rewards is not None}\n"
        s += f"global_auxiliary_rewards: {self.global_auxiliary_rewards is not None}\n"
        s += f"individual_rewards: {self.individual_rewards is not None}\n"
        s += f"individual_auxiliary_rewards: {self.individual_auxiliary_rewards is not None}\n"
        s += f"discrete_actions: {self.discrete_actions is not None}\n"
        s += f"discrete_action_cardinalities: {self.discrete_action_cardinalities}\n"
        s += f"continuous_actions: {self.continuous_actions is not None}\n"
        s += f"discrete_log_probs: {self.discrete_log_probs is not None}\n"
        s += f"continuous_log_probs: {self.continuous_log_probs is not None}\n"
        s += f"action_mask: {self.action_mask is not None}\n"
        s += f"action_mask_: {self.action_mask_ is not None}\n"
        s += f"terminated: {self.terminated is not None}\n"
        s += f"memory_weights: {self.memory_weights is not None}\n"

        s += f"obs: {self.obs}\n"
        s += f"obs_: {self.obs_}\n"
        s += f"state: {self.state}\n"
        s += f"state_: {self.state_}\n"
        s += f"global_rewards: {self.global_rewards}\n"
        s += f"global_auxiliary_rewards: {self.global_auxiliary_rewards}\n"
        s += f"individual_rewards: {self.individual_rewards}\n"
        s += f"individual_auxiliary_rewards: {self.individual_auxiliary_rewards}\n"
        s += f"discrete_actions: {self.discrete_actions}\n"
        s += f"continuous_actions: {self.continuous_actions}\n"
        s += f"discrete_log_probs: {self.discrete_log_probs}\n"
        s += f"continuous_log_probs: {self.continuous_log_probs}\n"
        s += f"action_mask: {self.action_mask}\n"
        s += f"action_mask_: {self.action_mask_}\n"
        s += f"terminated: {self.terminated}\n"
        s += f"memory_weights: {self.memory_weights}\n"
        return s

    def reset(self):
        self.idx = 0
        self.steps_recorded = 0
        self.episode_inds = None
        self.episode_lens = None

    def summarize_buffer(self):
        n_elem = (
            np.size(self.discrete_actions)
            + np.size(self.obs) * 2
            + np.size(self.global_rewards)
            + np.size(self.terminated)
        )
        if self.action_mask is not None:
            n_elem += np.size(self.action_mask) * 2
        er = ""
        if self.global_auxiliary_rewards is not None:
            er = "*2"
            n_elem += np.size(self.global_auxiliary_rewards)
        print(
            f"Buffer size: {self.steps_recorded} / {self.mem_size} steps. Current idx: {self.idx}. discrete_action_cardinalities: {self.discrete_action_cardinalities}, state: {self.obs.shape[1]}"
        )
        print(
            f"Total elements: actions {np.size(self.discrete_actions)} + states {np.size(self.obs)}*2 + rewards {np.size(self.global_rewards)}{er} + legality {np.size(self.action_mask)}*2 + done {np.size(self.terminated)} = {n_elem}"
        )

        if self.steps_recorded == 0:
            return

        start = self.idx % self.steps_recorded
        rs = []
        ex_rs = []
        rs.append(0)
        ex_rs.append(0)
        looping = True
        while looping:
            if start == self.idx - 1:
                looping = False
            rs[-1] += self.global_rewards[start]
            if self.global_auxiliary_rewards:
                ex_rs[-1] += self.global_auxiliary_rewards[start]

            if self.terminated[start] == 1:
                rs.append(0)
                if self.global_auxiliary_rewards:
                    ex_rs.append(0)

            start += 1
            start = start % self.steps_recorded
            # print(start)
            # print(self.steps_recorded)

        print(f"Episode Cumulative Rewards: {rs},\nExpert Rewards: {ex_rs}")

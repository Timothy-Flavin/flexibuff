Flexible Buffer supports numpy and torch tensor outputs formats,
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
```
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
```
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
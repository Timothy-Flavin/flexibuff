# Example single agent using flexibuff
# with torch dqn example modified from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import gym
from src.flexibuff import FlexibleBuffer

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1")
# Get number of actions from gym action space
n_actions = env.action_space.n  # type: ignore
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Set up the memory buffer for use with one agent,
# global reward and one discrete action output
memory = FlexibleBuffer(
    num_steps=10000,
    discrete_action_cardinalities=[2],
    path="./test_save/",
    name="all_attributes",
    n_agents=1,
    global_registered_vars={"global_rewards": (None, np.float32)},
    individual_registered_vars={
        "obs": ([n_observations], np.float32),
        "obs_": ([n_observations], np.float32),
        "discrete_actions": ([1], np.int64),
    },
)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


NUM_EPISODES = 800
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 3e-5

policy_net = DQN(n_observations, n_actions).to(device).float()
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
steps_done = 0


def select_action(state):
    state = torch.from_numpy(state)[None, :].to(device)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1).cpu().item()
    else:
        return (
            torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            .cpu()
            .item()
        )


def optimize_model():
    if memory.steps_recorded < BATCH_SIZE:
        return
    transitions = memory.sample_transitions(BATCH_SIZE, device=device, as_torch=True)
    term = transitions.terminated
    # print(transitions)
    if term is None:
        term = torch.zeros((BATCH_SIZE,), dtype=torch.bool, device=device)
    # print(transitions)
    # [0] because we are doing this for just the first agent because there is only one agent
    Q = policy_net(transitions.obs[0]).gather(1, transitions.discrete_actions[0])[:, 0]

    # print(
    #    f"Gamma: {GAMMA} * {policy_net(transitions.obs_[0]).max(1).values} * { (1 - transitions.terminated)} + {transitions.global_rewards}"
    # )
    with torch.no_grad():  # no need to track gradient for next Q value
        Q_NEXT = (
            GAMMA  # obs [0] because we are only using 1 agent again
            * policy_net(transitions.obs_[0]).max(1).values
            * (1 - term)  # Terminated and global rewards do
            + transitions.global_rewards  # not need to be [0] because they
        )  # are for all agents

    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(Q, Q_NEXT)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


ts = []
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get its state
    done = False
    state, info = env.reset()
    t = 0
    while not done:
        t += 1
        action = select_action(state)
        state_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store the transition in memory which accepts np arrays
        memory.save_transition(
            terminated=terminated,
            registered_vals={
                "global_rewards": reward,
                "obs": [state],
                "obs_": [state_],
                "discrete_actions": [[action]],
            },
        )

        # Move to the next state
        state = state_

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        policy_net_state_dict = policy_net.state_dict()
        if done:
            print(f"len: {t}")
            ts.append(t)
print("Complete")
plt.plot(ts)
plt.show()

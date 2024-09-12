import pettingzoo.butterfly.knights_archers_zombies_v10 as kaz
from flexibuff import FlexibleBuffer
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

NUM_KNIGHTS = 2
NUM_ARCHERS = 2
# Create the environment
env = kaz.parallel_env(
    spawn_rate=20,
    num_archers=NUM_ARCHERS,
    num_knights=NUM_KNIGHTS,
    max_zombies=10,
    max_arrows=10,
    killable_knights=True,
    killable_archers=True,
    pad_observation=True,
    line_death=False,
    max_cycles=900,
    vector_state=True,
    use_typemasks=False,
    sequence_space=False,
    # render_mode="human",
)
# Reset the environment
observations = env.reset()[0]
a1 = env.agents[0]
print(observations[a1])
original = env.agents.copy()

obs_size = np.array(observations[a1]).flatten().shape[0]

# Set up the memory buffer for use with one agent,
# global reward and one discrete action output
memory = FlexibleBuffer(
    num_steps=50000,
    obs_size=obs_size,
    discrete_action_cardinalities=[6],
    path="./test_save/",
    name="all_attributes",
    n_agents=NUM_ARCHERS + NUM_KNIGHTS,
    individual_reward=True,
    global_reward=False,
)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        self.float()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


NUM_EPISODES = 2000
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 40000
LR = 3e-5

policy_nets = [
    DQN(obs_size, 6).to(device),
    DQN(obs_size, 6).to(device),
]
optimizers = [
    optim.AdamW(policy_nets[0].parameters(), lr=LR, amsgrad=True),
    optim.AdamW(policy_nets[1].parameters(), lr=LR, amsgrad=True),
]
steps_done = 0


def ai_to_type(ai):
    if ai < NUM_ARCHERS:
        return 0
    else:
        return 1


def select_action(state, ai):
    state = torch.from_numpy(state)[None, :].float().to(device)
    # print(state)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    # print(eps_threshold)
    # steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_nets[ai](state).max(1).indices.view(1, 1).cpu().item()
    else:
        return (
            torch.tensor([[random.randint(0, 5)]], device=device, dtype=torch.long)
            .cpu()
            .item()
        )


def optimize_model():
    transitions = memory.sample_transitions(BATCH_SIZE, device=device, torch=True)
    for a in range(NUM_ARCHERS + NUM_KNIGHTS):
        if memory.steps_recorded < BATCH_SIZE:
            return

        Q = policy_nets[ai_to_type(a)](transitions.obs[a]).gather(
            1, transitions.discrete_actions[a]
        )[:, 0]

        # print(
        #    f"Gamma: {GAMMA} * {policy_net(transitions.obs_[0]).max(1).values} * { (1 - transitions.terminated)} + {transitions.global_rewards}"
        # )
        with torch.no_grad():  # no need to track gradient for next Q value
            Q_NEXT = (
                GAMMA  # obs [0] because we are only using 1 agent again
                * policy_nets[ai_to_type(a)](transitions.obs_[a]).max(1).values
                * (1 - transitions.terminated)  # Terminated and global rewards do
                + transitions.individual_rewards[a]  # not need to be [0] because they
            )  # are for all agents

        # Compute MSE loss
        criterion = nn.MSELoss()
        loss = criterion(Q, Q_NEXT)

        # Optimize the model
        optimizers[ai_to_type(a)].zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_nets[ai_to_type(a)].parameters(), 100)
        optimizers[ai_to_type(a)].step()


progress = {}
for a in original:
    progress[a] = []

for ep in range(NUM_EPISODES):
    if ep % 100 == 0 and ep != 0:
        env = kaz.parallel_env(
            spawn_rate=20,
            num_archers=NUM_ARCHERS,
            num_knights=NUM_KNIGHTS,
            max_zombies=10,
            max_arrows=10,
            killable_knights=True,
            killable_archers=True,
            pad_observation=True,
            line_death=False,
            max_cycles=900,
            vector_state=True,
            use_typemasks=False,
            sequence_space=False,
            render_mode="human",
        )
        # Reset the environment
        observations = env.reset()[0]
    else:
        env = kaz.parallel_env(
            spawn_rate=20,
            num_archers=NUM_ARCHERS,
            num_knights=NUM_KNIGHTS,
            max_zombies=10,
            max_arrows=10,
            killable_knights=True,
            killable_archers=True,
            pad_observation=True,
            line_death=False,
            max_cycles=900,
            vector_state=True,
            use_typemasks=False,
            sequence_space=False,
        )
        # Reset the environment
        observations = env.reset()[0]
    # Run an episode
    steps = 0
    print(f"Episoe: {ep}")
    reward_dict = {}
    for a in original:
        reward_dict[a] = 0
    observations = env.reset()[0]
    terminations = {agent: False for agent in env.agents}
    while env.agents:
        steps_done += 1
        steps += 1
        actions = {}
        for ai, a in enumerate(original):
            if a in env.agents:
                actions[a] = select_action(observations[a].flatten(), ai_to_type(ai))
        observations_, rewards, terminations, truncations, infos = env.step(actions)
        # print(
        #    f"obs: {observations_}, rew: {rewards}, term: {terminations}, trunc {truncations}"
        # )

        # Making observations for each agent
        obs = []
        obs_ = []
        term = True
        act = []
        rew = []
        for a in original:

            if a in env.agents:
                # print(observations[a].dtype)
                obs.append(observations[a].flatten())
                obs_.append(observations_[a].flatten())
                act.append(np.array([actions[a]], dtype=np.int32))
                term = term and terminations[a]
                rew.append(rewards[a])
            else:
                obs.append(np.zeros(obs_size, dtype=np.float64))
                obs_.append(np.zeros(obs_size, dtype=np.float64))
                act.append(np.zeros(1, dtype=np.int32))
                rew.append(0)
            reward_dict[a] += rewards[a]

        memory.save_transition(
            obs=obs,
            obs_=obs_,
            terminated=term,
            discrete_actions=act,
            continuous_actions=None,
            global_reward=None,
            global_auxiliary_reward=None,
            individual_rewards=rew,
            individual_auxiliary_rewards=None,
            action_mask=None,
            action_mask_=None,
            state=None,
            state_=None,
            discrete_log_probs=None,
            continuous_log_probs=None,
            memory_weight=1.0,
        )
        # input()
        observations = observations_

        if term:
            for a in original:
                progress[a].append(reward_dict[a])
            print(f"steps: {steps}]")
            print(reward_dict)
            for i in range(50):
                optimize_model()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
                -1.0 * steps_done / EPS_DECAY
            )
            print(eps_threshold)
            print(steps_done)
            break

    # Close the environment
env.close()


for a in original:
    plt.plot(progress[a])
plt.show()

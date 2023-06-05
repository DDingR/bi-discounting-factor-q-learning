import sys
# sys.path.append("c:/users/user/anaconda3/envs/dqn/lib/site-packages")
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def main():
    # train setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_NAME = "all 2.txt"
    ENV_NAME = "Pendulum-v1"

    advice_prob = 0.1

    train_agent_num = 3
    memory_num = 2

    memory_size = int(1e4)

    agent_params = [
        # eps, gamma
        [0.9,   0.99], # epsilon greedy
        [0,     0.1], # small gamma
        [0,     0.9], # large gamma
    ]

    action_list = np.array(range(-20,21)) / 10.0

    # constant 
    num_episodes = 1000
    n_observations = 3

    # agent and replay buffers define
    agent_list = []
    for i in range(train_agent_num):
        epsilon, gamma = agent_params[i]

        agent_list.append(
            DQNagent(n_observations, action_list, ENV_NAME, device, epsilon, gamma)
        )

    memory_list = []
    for i in range(memory_num):
        memory_list.append(ReplayMemory(memory_size))

    # list set up
    reward_list = []

    try:
        # ======================== TRAIN ===================================
        for i_episode in range(num_episodes):
            # initialization 
            reward_sum = [0, 0, 0]
            for agent in agent_list:
                agent.reset()

            for t in count():
                done0, reward0 = agent_list[0].train(memory_list[0], agent_list[0])
                
                if advice_prob > random.random():
                    done1, reward1 = agent_list[1].train(memory_list[1], agent_list[2])
                    done2, reward2 = agent_list[2].train(memory_list[1], agent_list[1])
                else:
                    done1, reward1 = agent_list[1].train(memory_list[1], agent_list[1])
                    done2, reward2 = agent_list[2].train(memory_list[1], agent_list[2])

                # reward_sum = [reward0, reward1, reward2] + reward_sum
                reward = [reward0, reward1, reward2]
                reward_sum = [reward_sum[i]+reward[i] for i in range(train_agent_num)]

                done = done1 and done2 and done0

                if done:
                    print(f"EPISODE {i_episode} \t| REWARD {np.round(reward_sum, 3)} \t| STEP {t}")
                    reward_list.append(reward_sum)
                    break
        # ======================== TRAIN ===================================
    finally:
        np.savetxt(TRAIN_NAME, reward_list, delimiter=',')

        print('Complete')
        plot_rewards(reward_list, show_result=True)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNagent():

    def __init__(self, n_observations, action_list, ENV_NAME, device, epsilon, gamma):
        self.TRAIN_NAME = 'TEST TRAIN'
        self.BATCH_SIZE = 128
        self.GAMMA = gamma
        self.TAU = 0.005
        self.LR = 1e-4

        self.device = device

        self.EPS_START = epsilon
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000

        self.action_list = action_list
        self.env = gym.make(ENV_NAME)

        self.n_actions = len(self.action_list)
        self.n_observations = n_observations

        self.state = None

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)  
            
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.steps_done = None

    def reset(self):
        self.steps_done = 0
        self.state, _ = self.env.reset()
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def train(self, memory, agent):
        self.Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

        state = self.state
        action_index = self.select_action()    # pi

        action = self.action_list[action_index]
        action = torch.tensor(action).view(1)

        observation, reward, terminated, truncated, _ = self.env.step(action) 


        reward = torch.tensor([reward], device=self.device, dtype=torch.float64)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action_index, next_state, reward)

        # Move to the next state
        self.state = next_state

        # Perform one step of the optimization (on the policy network)
        self.optimize_model(agent, memory)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        return done, reward.cpu().item()

    def select_action(self):
        sample = random.random()   # (0,1)
        if self.EPS_START != 0:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)    # 0.05 + (0.9-0.05) * 
        else:
            eps_threshold = 0
        self.steps_done += 1

        if sample > eps_threshold:    # if eps_threshold = 0 always 
            with torch.no_grad():
                tmp_action =  self.policy_net(self.state).max(1)[1]
                return tmp_action.view(1, 1)
        else:
            # tmp = np.array([env.action_space.sample()])
            tmp = np.random.choice([n for n in range(self.n_actions)])
            return torch.tensor(tmp, device=self.device, dtype=torch.long).view(1,1)

    def optimize_model(self, agent, memory):
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(self.BATCH_SIZE)

        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def plot_rewards(reward_list, show_result=False):
    plt.figure(1)
    reward_list = torch.tensor(reward_list, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_list) >= 100:
        means = reward_list.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

if __name__ == "__main__":
    main()

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from os.path import exists
from itertools import count
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from envs.simple_tag import SimpleTagCreator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

Networks = namedtuple('Networks',
                      ('policy_net', 'target_net', 'memory', 'optimizer'))

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    HIDDEN_LAYER_SIZE = 128

    # TODO: Change this to get env instead. And get from it the sizes
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, self.HIDDEN_LAYER_SIZE)
        self.layer2 = nn.Linear(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE)
        self.layer3 = nn.Linear(self.HIDDEN_LAYER_SIZE, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DecentralizedRunner():

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    def __init__(self, env):
        self.env = env
        self.steps_done = 0

        self.policies = {}
        for agent in env.possible_agents:
            # Get the size of the network
            n_obs = env.observation_space(agent).shape[0]
            n_act = env.action_space(agent).n

            # Create networks and helper classes
            policy_net = DQN(n_obs, n_act).to(device)
            if exists(f"{agent}.torch"):
                print("Found an existing model, loading...")
                policy_net.load_state_dict(torch.load(f"{agent}.torch"))

            target_net = DQN(n_obs, n_act).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)

            # Insert to the policies dict
            self.policies[agent] = Networks(policy_net, target_net, ReplayMemory(10000), optimizer)

    def compute_action(self, state, policy_id):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policies[policy_id].policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space(policy_id).sample()]], device=device, dtype=torch.long)

    @staticmethod
    def convert_from_tensor(actions):
        new_actions = {}
        for policy_id, action in actions.items():
            # Convert the action from tensor to number
            new_actions.update({policy_id: action.item()})
    
        return new_actions
    
    @staticmethod
    def convert_to_tensor(states):
        '''
        Convert state to pytorch tensors
        '''
        new_states = {}
        for agent, state in states.items():
            # Convert the state to tensor
            new_states.update({agent: torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)})

        return new_states

    def compute_actions(self, state, is_training=False):
        actions = {}
        for policy_id, _ in self.policies.items():
            actions.update({policy_id: self.compute_action(state[policy_id], policy_id)})
        
        return actions

    def optimize_model(self, policy_id):
        if len(self.policies[policy_id].memory) < self.BATCH_SIZE:
            return
        transitions = self.policies[policy_id].memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policies[policy_id].policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policies[policy_id].target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.policies[policy_id].optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policies[policy_id].policy_net.parameters(), 100)
        self.policies[policy_id].optimizer.step()

    def _train_policy(self, state, policy_id):
        pass 

    @staticmethod
    def add_reward(rewards_sum, reward):
        for policy, reward_value in reward.items():
            rewards_sum[policy] += reward_value

    def save_model(self):
        for policy, networks in self.policies.items():
            path = f"{policy}.torch"
            print(f"Saving model of {policy} at {path}")
            torch.save(networks.policy_net.state_dict() ,path)

    def train(self):
        num_episodes = 200
        episode_durations = []

        for i_episode in range(num_episodes):
            print(f"Running episode number {i_episode}")
            # Initialize the environment and get it's state
            state, info = env.reset()
            state = self.convert_to_tensor(state)
            episode_reward = {policy : 0 for policy in self.policies.keys()}

            for t in count():
                action = self.compute_actions(state, is_training=True)
                
                observation, reward, terminated, truncated, _ = env.step(self.convert_from_tensor(action))
                self.add_reward(episode_reward, reward)
                reward = self.convert_to_tensor(reward)
                done = all(terminated.values()) or all(truncated.values())

                if done:
                    next_state = None
                else:
                    next_state = self.convert_to_tensor(observation)

                for policy, networks in self.policies.items():
                    # Store the transition in memory
                    if next_state is not None:
                        networks.memory.push(state[policy], action[policy], next_state[policy], reward[policy])
                    else:
                        networks.memory.push(state[policy], action[policy], None, reward[policy])

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model(policy)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = networks.target_net.state_dict()
                    policy_net_state_dict = networks.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    networks.target_net.load_state_dict(target_net_state_dict)

                # Move to the next state
                state = next_state

                if done:
                    pprint(episode_reward)
                    episode_durations.append(sum(episode_reward.values()))
                    # plot_durations(episode_durations)
                    self.save_model()
                    break

        print('Complete')
        plot_durations(episode_durations, show_result=True)
        plt.ioff()
        plt.show()

env = SimpleTagCreator.create_env()
runner = DecentralizedRunner(env)
runner.train()



import math
import random
from os import makedirs, listdir
from os.path import exists, join, dirname
from collections import namedtuple, deque, Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .algo_creator import AlgoCreator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

Networks = namedtuple('Networks',
                      ('policy_net', 'target_net', 'memory', 'optimizer', 'iteration'))


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

class Controller():

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    RESULTS_PATH = "torch_results/"
    SAVE_FREQ = 10

    def __init__(self, config, env, env_name):

        self.env = env
        self.env_name = env_name
        self.steps_done = 0

        self.policy_mapping_fn = config.multiagent['policy_mapping_fn']

        self.policies = {}
        for agent in env.possible_agents:
            policy_name = self.policy_mapping_fn(agent, None, None)
            # If we already created a policy from the agent skip this part
            if policy_name in self.policies.keys():
                continue
            # Get the size of the network
            n_obs = env.observation_space(agent).shape[0]
            n_act = env.action_space(agent).n

            # Create networks and helper classes
            policy_net = DQN(n_obs, n_act).to(device)
            optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
            iteration = Counter()
            dir_name = self.load_checkpoint(policy_name)
            if dir_name is not None:
                saved_policy_path = join(self.RESULTS_PATH, self.env_name, dir_name, f"{policy_name}.torch")
                if exists(saved_policy_path):
                    print(f"Found an existing model at {saved_policy_path}, loading...")
                    checkpoint = torch.load(saved_policy_path)
                    policy_net.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    iteration['i'] = checkpoint['iteration']

            target_net = DQN(n_obs, n_act).to(device)
            target_net.load_state_dict(policy_net.state_dict())

            # Insert to the policies dict
            self.policies[policy_name] = Networks(policy_net, target_net, ReplayMemory(10000), optimizer, iteration)

    def load_checkpoint(self, policy_name):
        base_path = join(self.RESULTS_PATH, self.env_name)
        if not exists(base_path):
            return
        
        dirs = listdir(base_path)
        if not dirs:
            return
        
        # Returns the last checkpoint found based on the highest number
        checkpoints_found = sorted(filter(lambda dir: dir.startswith("checkpoint_"), dirs))
        return checkpoints_found[-1] if checkpoints_found else None

    def compute_single_action(self, state, policy_id, agent=None, is_training=False):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_action = self.policies[policy_id].policy_net(state).max(1)[1].view(1, 1)

        # If we are training the model we want to randomly choose an action once in a while
        if is_training:
            assert agent is not None, "agent must be provided when training"
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample < eps_threshold:
                policy_action = torch.tensor([[self.env.action_space(agent).sample()]], device=device, dtype=torch.long)

        return policy_action


    def compute_actions(self, state, policy_id, is_training=False):
        # Convert state to tensor
        state = self.convert_to_tensor(state)

        actions = {}
        for agent, obs in state.items():
            actions[agent] = self.compute_single_action(obs, policy_id, agent, is_training)

        return self.convert_from_tensor(actions)

    @staticmethod
    def convert_from_tensor(values):
        new_values = {}
        for policy_id, v in values.items():
            # Convert the v from tensor to value
            if isinstance(v, torch.Tensor):
                new_values.update({policy_id: v.item()})
            else:
                new_values.update({policy_id: v})

        return new_values

    @staticmethod
    def convert_to_tensor(values):
        '''
        Convert state to pytorch tensors
        '''
        new_values = {}
        for agent, v in values.items():
            # Convert the v to tensor if needed
            if not isinstance(v, torch.Tensor):
                dtype = torch.int64 if isinstance(v, int) else torch.float32
                new_values[agent] = torch.tensor(v, dtype=dtype, device=device).unsqueeze(0)
            else:
                new_values[agent] = v

        return new_values

    def compute_all_actions(self, state):
        observation_map = {policy:{} for policy in self.policies}
        for policy, observation in state.items():
            observation_map[self.policy_mapping_fn(policy, None, None)].update({policy: observation})
            
        actions = {}
        for policy, observation in observation_map.items():
            actions.update(self.compute_actions(observation, policy, True))    
            
        actions = self.convert_to_tensor(actions)

        # Converting a tensor back and forth changes the dimension
        # The reshape is needed to convert back to the original dimension
        for policy_id, tensor in actions.items():
            actions[policy_id] = tensor.reshape(-1, 1)
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

    def add_reward(self, rewards_sum, reward):
        for policy, reward_value in reward.items():
            rewards_sum[self.policy_mapping_fn(policy, None, None)] += reward_value

    def save_model(self):
        for policy, networks in self.policies.items():
            iteration = networks.iteration.get('i') or 0
            networks.iteration.update('i')
            if iteration % self.SAVE_FREQ != 0:
                continue
            path = join(self.RESULTS_PATH, self.env_name, f"checkpoint_{str(iteration).zfill(5)}", f"{policy}.torch")
            makedirs(dirname(path), exist_ok=True)
            print(f"Saving iteration {iteration} of {policy} at {path}")
            torch.save({
                'model_state_dict': networks.policy_net.state_dict(),
                'optimizer_state_dict': networks.optimizer.state_dict(),
                'iteration': iteration} ,path)

    def train(self):
        num_episodes = 2000

        for i_episode in range(num_episodes):
            print(f"Running episode number {i_episode}")
            # Initialize the environment and get it's state
            state, info = self.env.reset(seed=42)
            state = self.convert_to_tensor(state)
            episode_reward = {policy : 0 for policy in self.policies.keys()}
            
            done = False
            while not done:
                action = self.compute_all_actions(state)

                observation, reward, terminated, truncated, _ = self.env.step(self.convert_from_tensor(action))
                self.add_reward(episode_reward, reward)
                reward = self.convert_to_tensor(reward)
                done = all(terminated.values()) or all(truncated.values())

                if done:
                    next_state = None
                else:
                    next_state = self.convert_to_tensor(observation)

                for agent in self.env.possible_agents:
                    # Store the transition in memory
                    policy = self.policy_mapping_fn(agent, None, None)
                    network = self.policies[policy]
                    if next_state is not None:
                        network.memory.push(state[agent], action[agent], next_state[agent], reward[agent])
                    else:
                        network.memory.push(state[agent], action[agent], None, reward[agent])

                for agent in self.env.possible_agents:
                    policy = self.policy_mapping_fn(agent, None, None)
                    network = self.policies[policy]
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model(policy)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = network.target_net.state_dict()
                    policy_net_state_dict = network.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    network.target_net.load_state_dict(target_net_state_dict)

                # Move to the next state
                state = next_state

            print(episode_reward)
            self.save_model()

        print('Complete')

from ray.rllib.algorithms.dqn import DQNConfig

class CustomDQNCreator(AlgoCreator):

    def get_algo(self, config, env=None, env_name=""):
        return Controller(config, env, env_name)
    
    def get_algo_name(self):
        return "CustomDQN"
    
    def get_config(self, env_name):
        return DQNConfig()

    def train(self, config, env=None, env_name=""):
        self.get_algo(config, env, env_name).train()
#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import time
import argparse

from pathlib import Path

def validate_path(path: str):
    '''
    Check if the path exists
    '''
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist.")
    return path

parser = argparse.ArgumentParser(description="Train or evaluate an agent")
parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Choose 'train' or 'evaluate' mode")
parser.add_argument("--checkpoint-path", type=validate_path, help="Checkpoint path for evaluation")
parser.add_argument("--algo", type=str, choices=["ppo", "dqn", "custom_dqn"], default="custom_dqn", help="The algorithm to run on the env")
parser.add_argument("-e", "--env", type=str, choices=["simple_tag", "city_learn"], default="simple_tag", help="The environment to run")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-c", "--centralized", action='store_true', help="The algorithm will be centralized")
group.add_argument("-d", "--decentralized", action='store_true', help="The algorithm will be decentralized")

try:
    import argcomplete
    argcomplete.autocomplete(parser)
except ImportError:
    print("Please install argcomplete for auto completion")

import ray
from ray import tune

from multi_taxi.wrappers.petting_zoo_parallel import ParallelPettingZooEnvWrapper

from pettingzoo import ParallelEnv
from envs.env_creator import EnvCreator
from envs.simple_tag import SimpleTagCreator
from envs.city_learn import CityLearnCreator

from algorithms.ppo import PPOCreator
from algorithms.dqn import DQNCreator
from algorithms.my_dqn import CustomDQNCreator
from algorithms.algo_creator import AlgoCreator


class ParallelEnvRunner:

    def __init__(self, env_creator: EnvCreator, env: ParallelEnv, algorithm_name, config, is_centralized=True):
        ray.init()

        self.env = env
        self.env_name = env_creator.get_env_name()

        # Set policies based on centralized vs decentralized
        self.policies, self.policy_mapping_fn = env_creator.get_centralized() if is_centralized \
                                                else env_creator.get_decentralized()
        config.multi_agent(policies=self.policies, policy_mapping_fn=self.policy_mapping_fn)

        self.config = config
        self.algorithm_name = algorithm_name
        
        actual_env = self.create_env(config)
        tune.register_env(self.env_name, lambda config: ParallelPettingZooEnvWrapper(actual_env))
        
        # Set back the wrapped env
        self.env = actual_env


    def __del__(self):
        ray.shutdown()

    def create_env(self, config):
        '''
        This is a function called when registering a new env.
        '''
        return self.env

    def train(self):
        '''
        This is the function used to train the policy
        '''
        tune.run(
            self.algorithm_name,
            name=self.algorithm_name,
            stop={"timesteps_total": 5000000},
            checkpoint_freq=1,
            checkpoint_score_attr="episode_reward_mean",
            local_dir="ray_results/" + self.env_name,
            config=self.config.to_dict(),
            resume="AUTO"
        )


    def evaluate(self, algorithm, checkpoint_path: str = None, seed: int = 42):
        # Create an agent to handle the environment
        agent = algorithm.get_algo(self.config, env=self.env, env_name=self.env_name)
        if checkpoint_path is not None:
            agent.restore(checkpoint_path)

        # Setup the environment
        obs, _ = self.env.reset(seed=seed)
        self.env.render()

        observation_map = {policy:{} for policy in self.policies}
        reward_sum = 0
        i = 1

        while True:
            # Get actions from the policy
            for policy, observation in obs.items():
                observation_map[self.policy_mapping_fn(policy, None, None)].update({policy: observation})
            
            action_dict = {}
            for policy, observation in observation_map.items():
                action_dict.update(agent.compute_actions(observation, policy_id=policy))    
            
            # Step the environment with the chosen actions
            next_obs, rewards, term, trunc, _ = self.env.step(action_dict)
            
            # Update the episode reward
            reward_sum += sum(rewards.values())
            
            # Check if we need to stop the evaluation
            if all(term.values()):
                print("Termineting")
                break

            if all(trunc.values()):
                print("Truncating")
                break    

            obs = next_obs

            time.sleep(0.01)
            self.env.render()

            print(f"Step {i} - Total Reward: {reward_sum}")
            i += 1


def get_algorithm(algo_name: str) -> AlgoCreator:
    if algo_name == "ppo":
        return PPOCreator
    elif algo_name == "dqn":
        return DQNCreator
    elif algo_name == "custom_dqn":
        return CustomDQNCreator
    
def get_env(env_name: str) -> EnvCreator:
    if env_name == "simple_tag":
        return SimpleTagCreator
    elif env_name == "city_learn":
        return CityLearnCreator

if __name__ == "__main__":
    args = parser.parse_args()

    algorithm = get_algorithm(args.algo)
    env_creator = get_env(args.env)

    render_mode = 'human' if args.mode == 'evaluate' else None
    runner = ParallelEnvRunner(env_creator, env_creator.create_env(render_mode), 
                               algorithm.get_algo_name(), algorithm.get_config(env_creator.get_env_name()),
                               args.centralized)
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'evaluate':
        runner.evaluate(algorithm, args.checkpoint_path)

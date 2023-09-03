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
parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="dqn", help="The algorithm to run on the env")

try:
    import argcomplete
    argcomplete.autocomplete(parser)
except ImportError:
    print("Please install argcomplete for auto completion")

import ray
from ray import tune

from multi_taxi.wrappers.petting_zoo_parallel import ParallelPettingZooEnvWrapper

from pettingzoo import ParallelEnv
from envs.simple_tag import SimpleTagCreator

from algorithms.ppo import PPOCreator
from algorithms.dqn import DQNCreator
from algorithms.algo_creator import AlgoCreator

from supersuit.multiagent_wrappers.padding_wrappers import pad_observations_v0

class ParallelEnvRunner:

    def __init__(self, env_name: str, env: ParallelEnv, algorithm_name, config):
        ray.init()

        self.env = env
        self.env_name = env_name

        self.config = config
        self.algorithm_name = algorithm_name
        
        actual_env = self.create_env(config)
        actual_env = pad_observations_v0(actual_env)
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
        agent = algorithm(config=self.config)
        if checkpoint_path is not None:
            agent.restore(checkpoint_path)

        # Setup the environment
        obs, _ = self.env.reset(seed=seed)
        self.env.render()

        reward_sum = 0
        i = 1

        while True:
            # Get actions from the policy
            adversary_obs = {"adversary_0": obs["adversary_0"], "adversary_1": obs["adversary_1"]}
            agent_obs = {"agent_0": obs["agent_0"]}

            action_dict = agent.compute_actions(adversary_obs, policy_id="adversary")
            action_dict.update(agent.compute_actions(agent_obs, policy_id="agent"))
            
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
    

if __name__ == "__main__":
    args = parser.parse_args()

    algorithm = get_algorithm(args.algo)

    render_mode = 'human' if args.mode == 'evaluate' else None
    runner = ParallelEnvRunner(SimpleTagCreator.get_env_name(), SimpleTagCreator.create_env(render_mode), 
                               algorithm.get_algo_name(), algorithm.get_config(SimpleTagCreator.get_env_name()))
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'evaluate':
        runner.evaluate(algorithm.get_algo(), args.checkpoint_path)

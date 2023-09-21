from abc import ABC, abstractmethod

from ray import tune

class AlgoCreator(ABC):

    @abstractmethod
    def get_algo(config, env=None, env_name=""):
        pass

    @abstractmethod
    def get_algo_name():
        pass

    @abstractmethod
    def get_config(env_name):
        pass

    def train(self):
        tune.run(
            self.get_algo_name(),
            name=self.get_algo_name(),
            stop={"timesteps_total": 5000000},
            checkpoint_freq=1,
            checkpoint_score_attr="episode_reward_mean",
            local_dir="ray_results/" + self.env_name,
            config=self.config.to_dict(),
            resume="AUTO"
        )
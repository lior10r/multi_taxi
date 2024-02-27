from abc import ABC, abstractmethod

from ray import tune

class AlgoCreator(ABC):

    @abstractmethod
    def get_algo(self, config, env=None, env_name=""):
        pass

    @abstractmethod
    def get_algo_name(self):
        pass

    @abstractmethod
    def get_config(self, env_name):
        pass

    def train(self, config, env=None, env_name=""):
        tune.run(
            self.get_algo_name(),
            name=self.get_algo_name(),
            stop={"timesteps_total": 5000000},
            checkpoint_freq=1,
            checkpoint_score_attr="episode_reward_mean",
            local_dir="ray_results/" + env_name,
            config=config.to_dict(),
            resume="AUTO"
        )
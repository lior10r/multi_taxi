from typing import Optional

from ray.rllib.env import ParallelPettingZooEnv


class ParallelPettingZooEnvWrapper(ParallelPettingZooEnv):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info or {}
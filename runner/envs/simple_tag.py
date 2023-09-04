from .env_creator import EnvCreator

from pettingzoo.mpe import simple_tag_v3

class SimpleTagCreator(EnvCreator):

    ENV_NAME = "simple_tag"

    @staticmethod
    def get_env_name():
        return "simple_tag"

    @staticmethod
    def create_env(render_mode=None):
        # using the PettingZoo parallel API here
        return simple_tag_v3.parallel_env(num_good=1, num_adversaries=2, num_obstacles=0, max_cycles=256, continuous_actions=False, render_mode=render_mode)

    @staticmethod
    def get_centralized():
        return {'agent', 'adversary'},  \
                lambda name, episode, worker, **kwargs: 'adversary' if 'adversary' in name else 'agent'   

    @staticmethod
    def get_decentralized():
        return {'agent_0', 'adversary_0', 'adversary_1'},  \
                lambda name, episode, worker, **kwargs: name   

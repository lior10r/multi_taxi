from .env_creator import EnvCreator

import os

import pandas as pd
import gymnasium as gym

from functools import lru_cache
from pettingzoo import ParallelEnv

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import TabularQLearningActionWrapper


def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity consumption, cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """

    kpis = env.evaluate()

    # names of KPIs to retrieve from evaluate function
    kpi_names = [
        'electricity_consumption', 'cost', 'carbon_emissions',
        'average_daily_peak', 'ramping', '1 - load_factor'
    ]
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()

    # round up the values to 3 decimal places for readability
    kpis['value'] = kpis['value'].round(3)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis


class CityLearnPettingZooWrapper(ParallelEnv):
    metadata = {"render_modes": [], "is_parallelizable": True}

    def __init__(self, env):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.citylearnenv = env
        if env.central_agent:
            self.possible_agents = ["city_0"]
        else:
            self.possible_agents = [f'building_{r}' for r in range(len(self.citylearnenv.buildings))]


    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @lru_cache(maxsize=1000)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        agent_num = int(agent.split('_')[1])
        obs = self.citylearnenv.observation_space[agent_num]
        # Convert gym box to gymnasium to work with rllib
        return gym.spaces.Box(obs.low, obs.high, obs.shape, obs.dtype, obs.seed()[0])

    @lru_cache(maxsize=1000)
    def action_space(self, agent):
        agent_num = int(agent.split('_')[1])
        act = self.citylearnenv.action_space[agent_num]
        return gym.spaces.Discrete(act.n, act.seed()[0])



    def render(self, mode="human"):
        print(get_kpis(self.citylearnenv))

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        all_obs = self.citylearnenv.reset()
        observations = {agent: all_obs[i] for i, agent in enumerate(self.agents) }
        return observations, {}

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            actions_list = [acs.sample() for acs in self.citylearnenv.action_space] 
            print("Empty actions array provided, using randomly sampled actions")
        else:
            actions_list = [actions[agent] for agent in self.agents]

        all_obs, all_rew, done, all_info = self.citylearnenv.step(actions_list)

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {a: r for a, r in zip(self.agents, all_rew)}

        self.num_moves += 1
        dones = {agent: done for agent in self.agents}

        observations = {agent: all_obs[i] for i, agent in enumerate(self.agents) }
        infos = {agent: all_info for agent in self.agents}

        truncateds = {name: False for name in self.agents}
        if done:
            self.agents = [] # Required feature in pettingzoo


        return observations, rewards, dones, truncateds, infos


class CityLearnCreator(EnvCreator):

    ENV_NAME = "city_learn"

    @staticmethod
    def get_env_name():
        return "city_learn"

    @staticmethod
    def create_env(render_mode=None):
        schema_path = os.path.join(os.path.dirname(__file__), "city_learn/schema.json")
        env = CityLearnEnv(schema_path, central_agent=True, simulation_end_time_step=1000)
        # For a discrete action space
        env = TabularQLearningActionWrapper(env)
        return CityLearnPettingZooWrapper(env)

    @staticmethod
    def get_centralized():
        return {'city_0'},  \
                lambda name, episode, worker, **kwargs: name   

    @staticmethod
    def get_decentralized():
        return {'agent_0', 'adversary_0', 'adversary_1'},  \
                lambda name, episode, worker, **kwargs: name   

'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random

# Other imports.
import gymnasium as gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, env_name='CartPole-v1', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = NormalizedEnv(gym.make(env_name))
        self.render = render
        # Gymnasium reset returns (obs, info)
        init_obs, _init_info = self.env.reset()
        MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func, init_state=GymState(init_obs))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, terminated, truncated, info = self.env.step(action)
        is_terminal = bool(terminated) or bool(truncated)

        if self.render:
            self.env.render()

        self.next_state = GymState(obs, is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        # Maintain Gymnasium semantics; ignore info for MDP init here.
        self.env.reset()

    def seed(self, seed=None):
        """Seed the environment and its action space under Gymnasium semantics."""
        try:
            self.env.reset(seed=seed)
        except TypeError:
            pass
        try:
            self.env.action_space.seed(seed)
        except Exception:
            pass

    def __str__(self):
        return "gym-" + str(self.env_name)

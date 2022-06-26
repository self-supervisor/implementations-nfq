"""
Modified version of classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
# flake8: noqa
import math
from typing import Callable, List, Tuple

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding
from gym.envs.classic_control import MountainCarEnv


class MountainCarHintEnv(MountainCarEnv):
    def __init__(self):
        super().__init__()

    def get_goal_pattern_set(self, size: int = 100):
        """Use hint-to-goal heuristic to clamp network output.

        Parameters
        ----------
        size : int
            The size of the goal pattern set to generate.

        Returns
        -------
        pattern_set : tuple of np.ndarray
            Pattern set to train the NFQ network.

        """
        goal_state_action_b = [
            np.array(
                [
                    np.random.uniform(0.5, 0.6),
                    np.random.uniform(-0.07, 0.07),
                    np.random.randint(3),
                ]
            )
            for _ in range(size)
        ]
        goal_target_q_values = np.zeros(size)

        return goal_state_action_b, goal_target_q_values

    def generate_rollout(
        self, get_best_action: Callable = None, render: bool = False
    ) -> List[Tuple[np.array, int, int, np.array, bool]]:
        """
        Generate rollout using given action selection function.

        If a network is not given, generate random rollout instead.

        Parameters
        ----------
        get_best_action : Callable
            Greedy policy.
        render: bool
            If true, render environment.

        Returns
        -------
        rollout : List of Tuple
            Generated rollout.
        episode_cost : float
            Cumulative cost throughout the episode.

        """
        rollout = []
        episode_cost = 0
        obs = self.reset()
        done = False
        count = 0
        while not done:
            count += 1
            if get_best_action:
                action = get_best_action(obs)
            else:
                action = self.action_space.sample()

            next_obs, cost, done, info = self.step(action)
            rollout.append((obs, action, cost, next_obs, done))
            episode_cost += cost
            obs = next_obs

            if render:
                self.render()

            if count >= 1000:
                break

        return rollout, episode_cost

#-----------------------------------
# Author: Sneha Reddy Aenugu
# Description: 10x10 Gridworld domain
#----------------------------------

import numpy as np
import math
import argparse
from multiprocessing import Pool
from functools import partial
import random
import pickle
from matplotlib import pyplot as plt

action_set = {0:"AU", 1:"AD", 2:"AL", 3:"AR"}
improvement_array = []

class GridWorld687:
    def __init__(self):
        current_state = None;
        current_action = None
        self.array = np.concatenate((np.ones(80), np.ones(5) * 2, np.ones(5) * 3, np.ones(10) * 4))

    def transition_function(self, state, action): # 执行动作
        new_state = None
        reward = None
        if action == "U":   # 上行一步
            if state < 11:
                new_state = state
            else:
                new_state = state - 10

        elif action == "D": # 下行一步
            if state > 90:
                new_state = state
            else:
                new_state = state + 10

        elif action == "L": # 左行一步
            if state % 10 == 1:
                new_state = state
            else:
                new_state = state - 1

        elif action == "R": # 右行一步
            if state % 10 == 0:
                new_state = state
            else:
                new_state = state + 1

        else:
            new_state = state
        
        return new_state

    def reward_function(self, state):
        if state == 100:
            reward = 10
        else:
            reward = 0

        return reward

    def action_from_attempt(self, attempt): # 决策
        '''
        智能体朝预定方向移动的概率为0.8 (1)
        智能体偏离预定方向左转的概率为0.05 (2)
        智能体偏离预定方向右转的概率为0.05 (3)
        智能体不执行动作的概率为0.1 (4)
        '''
        # res = np.random.choice(np.arange(1, 5), p=[0.8, 0.05, 0.05, 0.1])
        res = self.array[random.randint(0, len(self.array) - 1)]

        if attempt == "AU":   # 预定方向上行
            if res == 1:
                action = "U"
            elif res == 2:
                action = "L"
            elif res == 3:
                action = "R"
            else:
                action = "N"

        elif attempt == "AD": # 预定方向下行
            if res == 1:
                action = "D"
            elif res == 2:
                action = "R"
            elif res == 3:
                action = "L"
            else:
                action = "N"

        elif attempt == "AL": # 预定方向左行
            if res == 1:
                action = "L"
            elif res == 2:
                action = "D"
            elif res == 3:
                action = "U"
            else:
                action = "N"

        elif attempt == "AR": # 预定方向右行
            if res == 1:
                action = "R"
            elif res == 2:
                action = "U"
            elif res == 3:
                action = "D"
            else:
                action = "N"

        return action

    def get_discounted_returns(self, rewards, gamma=0.9): # 计算折扣奖励
        discounted_reward = 0

        for i in range(len(rewards)):
            discounted_reward += math.pow(gamma, i) * rewards[i]

        return discounted_reward

    def run_one_episode_random_policy(self, initial_state): # 运行一回合随机策略
        states = []
        rewards = []
        actions = []
        state = initial_state

        while state != 23:
            attempt_action = action_set[np.random.choice(np.arange(0, 4), 1)[0]]
            action = self.action_from_attempt(attempt_action)
            new_state = self.transition_function(state, action)
            reward = self.reward_function(new_state)
            states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            state = new_state

        return states, rewards, actions

    def run_epsilon_greedy_policy(self, epsilon, q_func): # 运行回合贪婪策略
        states = []
        rewards = []
        actions = []
        state = 1
        count = 0
        action_ind = np.argmax(q_func[state])

        while state != 23 and count < 300:
            prob = (epsilon / 4) * np.ones(4)
            prob[action_ind] = (1 - epsilon) + (epsilon / 4)
            attempt = action_set[int(np.random.choice(4, 1, p=prob))]
            action = self.action_from_attempt(attempt)
            new_state = self.transition_function(state, action)
            new_action_ind = np.argmax(q_func[new_state])
            reward = self.reward_function(new_state)
            rewards.append(reward)
            state = new_state
            action_ind = new_action_ind
            count += 1

        return rewards

    def get_action_from_state(self, state):
        if state % 10 != 0:
            action = 'AR'
        else:
            action = 'AD'

        return action

    def run_given_policy(self, initial_state):
        states = []
        rewards = []
        actions = []
        state_action_pair = { 1: 'AR', 2: 'AR', 3: 'AR', 4: 'AR', 5: 'AD',
                              6: 'AR', 7: 'AR', 8: 'AR', 9: 'AR', 10: 'AD',
                              11: 'AU', 12: 'AU', 13: 'AR', 14: 'AD', 15: 'AU',
                              16: 'AU', 17: 'AR', 18: 'AD', 19: 'AU', 20: 'AU',
                              21: 'AR', 22: 'AR'
                            }

        state = initial_state

        while state != 100:
            action = self.action_from_attempt(self.get_action_from_state(state))
            new_state = self.transition_function(state, action)
            reward = self.reward_function(new_state)
            states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            state = new_state

        return states, rewards, actions
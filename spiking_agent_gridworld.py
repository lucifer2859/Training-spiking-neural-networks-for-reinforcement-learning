#-------------------------------------
# Author: Sneha Reddy Aenugu
# Description: Solving Gridworld with spiking agent actor-critic
# PGCN learning rule for a network of GLM spiking agents
#------------------------------------

import numpy as np
import math
import argparse
import random
import itertools
import pdb
import pickle
from gridworld import GridWorld687
import copy
np.set_printoptions(precision=2)

action_set = {0:"AU", 1:"AD", 2:"AL", 3:"AR"}
improvement_array = []


# Class for actor-critic module
class ActorCritic():
    def __init__(self, epsilon, alpha, lda):
        self.epsilon = 0.01 # 贪婪随机概率
        self.alpha = alpha # 学习率
        self.gamma = 0.9 # 折扣因子
        self.lda = lda # Lambda
        self.sigma = 10
        self.gw = GridWorld687()
        self.actors = [SpikingActor() for i in range(10)]

    def tabular_softmax(self, policy):
        softmax_policy = [np.exp(policy[i, :]) / sum(np.exp(policy[i])) for i in range(policy.shape[0])]
        
        return softmax_policy

    def e_greedy_policy(self, action_ind):
        prob = (self.epsilon / 4) * np.ones(4)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon / 4)
        
        return prob

    def softmax_selection(self, qvalues, sigma):
        eps = 1e-5
        qvalues = qvalues + eps
        prob = np.exp(sigma * qvalues) / sum(np.exp(sigma * qvalues))
        
        return prob

    def run_actor_critic(self, num_episodes):
        returns = []
        weights_outer = []
        weights_inner = []
        theta = np.random.rand(101, 4) # 101x4
        v_f = 10 * np.random.rand(101) # 101
        alpha = 0.1
        
        for i in range(num_episodes):
            rewards = []
            states = []
            states.append(0)
            state = 1
            e_theta = np.zeros_like(theta) # 101x4
            e_v = np.zeros_like(v_f) # 101
            gamma = 0.9
            count = 0

            while state != 100 and count < 300:
                # Act using actor
                actions = np.zeros((len(self.actors), 4)) # 10x4
                for k in range(len(self.actors)):
                    yo = self.actors[k].forward(state, count)
                    actions[k] = yo

                action_index = np.argmax(np.sum(actions, axis=0)) # 获取动作索引

                prob = self.e_greedy_policy(action_index) # e_贪婪概率
                attempt = int(np.random.choice(4, 1, p=prob)) # 基于e_贪婪概率获取尝试索引

                action = self.gw.action_from_attempt(action_set[attempt]) # 基于尝试获取动作
                if action != 'N':
                    action_ind = list(action_set.keys())[list(action_set.values()).index('A' + action)]
                else:
                    action_ind = attempt

                new_state = self.gw.transition_function(state, action) # 状态转换
                reward = self.gw.reward_function(new_state) # 获取奖励

                # Critic update
                e_v *= gamma * self.lda # 资格迹衰减
                e_v[state - 1] += 1 # 累积迹
                delta_t = reward + gamma * v_f[new_state - 1] - v_f[state - 1] # 单步时序差分形式的优势函数估计
                v_f += alpha * delta_t * e_v # 更新状态价值函数

                # Actor update
                for k in range(len(self.actors)):
                    self.actors[k].update_weights(delta_t, state, action_ind, np.mean(rewards[-5:])) # 更新执行者权重

                state = new_state
                count += 1
                rewards.append(reward)
                states.append(new_state)

            discounted_return = self.gw.get_discounted_returns(rewards) # 获取折扣奖励
            returns.append(discounted_return)
            
            if i % 20 == 0 or i == 99:
                print("Discounted Return after %s episodes: %s" % (i, discounted_return))

        return returns


# Generalized Linear Model of a Spiking Neuron
# Linear-nonlinear-Poisson cascade neuron model
class SpikingActor():
    def __init__(self):
        self.inputs = 3
        self.hidden = 2
        self.outputs = 4 # 输出层有4个合作者，每个合作者对应gridworld领域的动作，并且合作者的活动被编码在单个脉冲中

        self.ih = np.random.rand(4, self.hidden, self.inputs, 3) # 4x2x3x3
        self.ho = np.random.rand(self.outputs, self.hidden, 3) # 4x2x3
        self.vih = np.zeros((4, self.hidden, self.inputs, 3)) # 4x2x3x3
        self.vho = np.zeros((self.outputs, self.hidden, 3)) # 4x2x3

        self.input_map = self.state_coding() # 120x3x2

    def state_coding(self):
        combns = list(itertools.combinations(range(5), r=2)) # [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        maps = list(itertools.combinations(combns, r=3)) # 120x3x2

        return maps

    def input_coding(self, state):
        state_code = -1 * np.ones((3, 5)) # 3x5
        map_val = self.input_map[state] # 3x2
        state_code[0, map_val[0]] = 1
        state_code[1, map_val[1]] = 1
        state_code[2, map_val[2]] = 1
        
        return state_code # 3x5

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-2 * x))

    def forward(self, state, count):
        '''
        每个状态可以用二元值（1/-1）的S×K向量表示，其中S是空间因素，K是时间因素
        这里S为编码刺激的神经元数量，K为每个神经元的脉冲序列长度（以离散时间容器测量）
        gridworld的100个状态由3个神经元编码，其脉冲序列长度为5
        '''
        inputs = self.input_coding(state) # 3x5
 
        ph = np.zeros((4, self.hidden, 3)) # 4x2x3
        yh = np.zeros((4, self.hidden, 3)) # 4x2x3
        po = np.zeros(self.outputs) # 4
        yo = np.zeros(self.outputs) # 4
        dih = np.zeros((4, self.hidden, self.inputs, 3)) # 4x2x3x3
        dho = np.zeros((self.outputs, self.hidden, 3)) # 4x2x3

        for i in range(4):
            for j in range(self.hidden):
                for k in range(self.inputs):
                    ph[i, j] = self.sigmoid(np.convolve(inputs[k], self.ih[i, j, k])[2:5]) # 任何时刻的发放条件概率
                    yh[i, j] = (ph[i, j] > np.random.rand(3)).astype(int) # Poisson
                    deriv = yh[i, j] * (1 - ph[i, j]) + (1 - yh[i, j]) * (-ph[i, j])
                    dih[i, j, k, 0] = np.sum(inputs[k, 2:5] * deriv)
                    dih[i, j, k, 1] = np.sum(inputs[k, 1:4] * deriv)
                    dih[i, j, k, 2] = np.sum(inputs[k, 0:3] * deriv)
                    yh[i, j] = 2 * yh[i, j] - 1 # 将0/1编码转化为-1/1形式

        for i in range(4):
            for j in range(self.hidden):
                po[i] = self.sigmoid(np.matmul(yh[i, j], self.ho[i, j]))
                yo[i] = (po[i] > np.random.rand(1)).astype(int)
                deriv = yo[i] * (1 - po[i]) + (1 - yo[i]) * (-po[i])
                dho[i, j] = yo[i] * (1 - po[i]) * yh[i, j] + (-1) * (1 - yo[i]) * po[i] * yh[i, j]
                yo[i] = 2 * yo[i] - 1 # 将0/1编码转化为-1/1形式
        
        self.yo = yo
        self.yh = yh
        self.dih = dih
        self.dho = dho

        return yo

    def update_weights(self, tderror, state, action, count):
        if count < 0.4:
            alpha = 0.01
        else:
            alpha = 0.001
        self.beta = 0

        for i in range(4):
            if i == action and self.yo[i] > 0:
                dv = self.dih[i].copy()
                self.ih[i] += alpha * tderror * dv
            elif i == action and self.yo[i] < 0:
                self.ih[i] -= alpha * tderror * self.dih[i]
            elif i != action and self.yo[i] < 0:
                self.ih[i] += alpha * tderror * self.dih[i]
            elif i != action and self.yo[i] > 0:
                self.ih[i] -= alpha * tderror * self.dih[i]

        for i in range(4):
            if i == action and self.yo[i] > 0:
                dv = self.dho[i].copy()
                self.ho[i] += alpha * tderror * dv
            elif i == action and self.yo[i] < 0:
                self.ho[i] -= alpha * tderror * self.dho[i]
            elif i != action and self.yo[i] < 0:
                self.ho[i] += alpha * tderror * self.dho[i]
            elif i != action and self.yo[i] > 0:
                self.ho[i] -= alpha * tderror * self.dho[i]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='algorithm', default='sarsa')
    parser.add_argument('--selection', dest='selection', default='egreedy')
    parser.add_argument('--num_trials', dest='num_trials', default=1)
    parser.add_argument('--num_episodes', dest='num_episodes', default=1000)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--num', dest='num', default=1)

    args = parser.parse_args()

    rewards_trials = []

    if args.selection == 'egreedy':
        step_size = 0.1 
        epsilon = 0.01
        lda = 0.7
    else:
        step_size = 0.1 
        epsilon = 0.1

    step_size = 0.1

    for i in range(int(args.num_trials)):
        print('Trial:', i)
        td_cp = ActorCritic(epsilon=epsilon, alpha=step_size, lda=lda)
        rewards = td_cp.run_actor_critic(int(args.num_episodes))

        rewards_trials.append(rewards)

    print("Maximum reward reached at the end of 100 episodes : ", np.mean(rewards_trials, axis=0)[-1] )

    f = open('rewards_' + str(args.num) + '.pkl', 'wb')
    pickle.dump(rewards_trials, f)
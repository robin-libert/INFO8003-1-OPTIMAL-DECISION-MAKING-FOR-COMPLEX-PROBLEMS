# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:15:27 2020

@author: robin
"""

import random

class Agent:

    def __init__(self, domain):
        self.alpha = 0.05
        self.epsilon = 0.25
        self.initial_state = (3,0)
        self.current_state = (3,0)
        self.domain = domain
        self.Q = {}#initialize Q
        for i in self.domain.state_space:
            for state in i:
                for action in self.domain.action_space:
                    self.Q[(state, action)] = 0

    def train(self, n):
        """
        Train the agent over n episodes and update Q.
        First protocol.
        """
        for i in range(n):
            trajectory = self.explore(1000)
            t = ((len(trajectory)-1)/3)
            k = 0
            while k < t:
                k_index = k
                xk = trajectory[k_index * 3]
                uk = trajectory[k_index * 3 + 1]
                rk = trajectory[k_index * 3 + 2]
                xkplus1 = trajectory[k_index * 3 + 3]
                self.Q[(xk,uk)] = (1-self.alpha) * self.Q[(xk,uk)]
                maxi = 0
                for u in self.domain.action_space:
                    temp = self.Q[(xkplus1, u)]
                    if temp > maxi:
                        maxi = temp
                self.Q[(xk,uk)] += self.alpha * (rk + self.domain.discount_factor * maxi)
                k += 1
            maxi = 0
            for j in self.domain.state_space:
                for state in j:
                    temp = abs(self.domain.optimal_J[state] - self.Q[(state, self.domain.optimal_policy[state])])
                    if temp > maxi:
                        maxi = temp
            print(maxi)

    def train2(self, n):
        """
        Train the agent over n episodes and update Q.
        Second protocol.
        """
        for i in range(n):
            trajectory = self.explore(1000)
            t = ((len(trajectory)-1)/3)
            k = 0
            while k < t:
                if k > 0:
                    self.alpha = 0.8*self.alpha
                k_index = k
                xk = trajectory[k_index * 3]
                uk = trajectory[k_index * 3 + 1]
                rk = trajectory[k_index * 3 + 2]
                xkplus1 = trajectory[k_index * 3 + 3]
                self.Q[(xk,uk)] = (1-self.alpha) * self.Q[(xk,uk)]
                maxi = 0
                for u in self.domain.action_space:
                    temp = self.Q[(xkplus1, u)]
                    if temp > maxi:
                        maxi = temp
                self.Q[(xk,uk)] += self.alpha * (rk + self.domain.discount_factor * maxi)
                k += 1
            maxi = 0
            for j in self.domain.state_space:
                for state in j:
                    temp = abs(self.domain.optimal_J[state] - self.Q[(state, self.domain.optimal_policy[state])])
                    if temp > maxi:
                        maxi = temp
            print(maxi)

    def train3(self, n):
        """
        Train the agent over n episodes and update Q.
        Third protocol.
        """
        for i in range(n):
            trajectory = self.explore(1000)
            t = ((len(trajectory)-1)/3)
            k = 0
            buffer = []
            while k < t:
                k_index = k
                xk = trajectory[k_index * 3]
                uk = trajectory[k_index * 3 + 1]
                rk = trajectory[k_index * 3 + 2]
                xkplus1 = trajectory[k_index * 3 + 3]
                buffer.append((xk, uk, rk, xkplus1))
                for i in range(10):
                    index = random.randint(0, len(buffer)-1)
                    xk = buffer[index][0]
                    uk = buffer[index][1]
                    rk = buffer[index][2]
                    xkplus1 = buffer[index][3]
                    self.Q[(xk,uk)] = (1-self.alpha) * self.Q[(xk,uk)]
                    maxi = 0
                    for u in self.domain.action_space:
                        temp = self.Q[(xkplus1, u)]
                        if temp > maxi:
                            maxi = temp
                    self.Q[(xk,uk)] += self.alpha * (rk + self.domain.discount_factor * maxi)
                k += 1
            maxi = 0
            for j in self.domain.state_space:
                for state in j:
                    temp = abs(self.domain.optimal_J[state] - self.Q[(state, self.domain.optimal_policy[state])])
                    if temp > maxi:
                        maxi = temp
            print(maxi)

    def move(self):
        """
        The agent use this function to move according to an epsilon greedy policy.
        Update the current state of the agent.
        Return the action the agent chose.
        """
        n = random.random()
        if n < self.epsilon:#random
            action = self.domain.action_space[random.randint(0,3)]
            self.current_state = self.domain.move(self.current_state, action)
            return action
        else:#select best action
            action = self.domain.action_space[0]
            r = self.domain.reward_signal(self.domain.move(self.current_state, action))
            for u in self.domain.action_space:
                r_temp = self.domain.reward_signal(self.domain.move(self.current_state, u))
                if r_temp > r:
                    r = r_temp
                    action = u
            self.current_state = self.domain.move(self.current_state, action)
            return action

    def explore(self, t):
        """
        The agent explore the domain according to an epsilon greedy policy.
        Return the trajectory of size t.
        """
        ht = []
        self.current_state = self.initial_state
        ht.append(self.current_state)
        for i in range(t):
            action = self.move()
            arrivalState = self.current_state
            reward = self.domain.reward_signal(arrivalState)
            ht.append(action)
            ht.append(reward)
            ht.append(arrivalState)
        return ht

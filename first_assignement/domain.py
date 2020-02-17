# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:53:10 2020

@author: robin
"""
import random

class Domain:
    def __init__(self):
        self.g = [[-3,1,-5,0,19],
                  [6,3,8,9,10],
                  [5,-8,4,1,-8],
                  [6,-9,4,19,-5],
                  [-20,-17,-4,-3,9]]
        self.n = 5
        self.m = 5
        #0 for deterministic setting of the domain
        #1 for stochastic setting of the domain
        self.setting = 0
        self.state_space = [[(x,y) for y in range(self.n)] for x in range(self.m)]
        self.action_space = [(1,0),(-1,0),(0,1),(0,-1)]
        self.discount_factor = 0.99
        #markov decision process
        self.r = {}#r(x,u)
        self.p = {}#p(x'|x,u)
        self.Q = {}
        self.optimal_J = {}#JN(x) following an optimal policy
        self.optimal_policy = {}
        self.update()


    def move(self, state, action, setting=None):
        if setting == None:
            setting = self.setting
        if setting == 0:#deterministic setting
            return (min(max(state[0]+action[0],0),self.n-1), min(max(state[1]+action[1],0),self.m-1))
        else:#stochastic setting
            if random.uniform(0,1) <= 1-0.5:
                return (min(max(state[0]+action[0],0),self.n-1), min(max(state[1]+action[1],0),self.m-1))
            else:
                return (0,0)

    def reward_signal(self, newState):
    	reward = self.g[newState[0]][newState[1]]
    	return reward

    def update_r(self):
        """
        Compute the true value of r(x,u)
        """
        #initialize r(x,u)
        for i in self.state_space:
            for state in i:
                for action in self.action_space:
                    self.r[(state, action)] = 0.
        if self.setting == 0:#deterministic setting
            for i in self.state_space:
                for state in i:
                    for action in self.action_space:
                        self.r[(state, action)] = self.reward_signal(self.move(state, action))
        else:#stochastic setting
            for i in self.state_space:
                for state in i:
                    for action in self.action_space:
                        self.r[(state, action)] = (self.reward_signal(self.move(state, action, setting=0)) + self.reward_signal((0,0)))/2.

    def update_p(self):
        """
        Compute the true value of p(x'|x,u)
        """
        #initialize p(x'|x,u)
        for i in self.state_space:
            for state in i:
                for j in self.state_space:
                    for newState in j:
                        for action in self.action_space:
                            self.p[(state, action, newState)] = 0.
        for i in self.state_space:
            for state in i:
                for j in self.state_space:
                    for newState in j:
                        for action in self.action_space:
                            if self.setting == 0:
                                if self.move(state, action, setting=0) == newState:
                                    self.p[(state, action, newState)] = 1.
                            else:
                                if self.move(state, action, setting=0) == newState and newState != (0,0):
                                    self.p[(state, action, newState)] = 0.5
                                elif self.move(state, action, setting=0) == newState and newState == (0,0):
                                    self.p[(state, action, newState)] = 1.
                                elif self.move(state, action, setting=0) != newState and newState == (0,0):
                                    self.p[(state, action, newState)] = 0.5

    def memoize(f):
        """
        To optimize the Q function
        """
        memo = {}
        def helper(s,a,b,c):
            if (a,b,c) not in memo:
                memo[(a,b,c)] = f(s,a,b,c)
            return memo[(a,b,c)]
        return helper

    @memoize
    def My_Q(self, state, action, N):
        if N == 0:
            return 0
        else:
            mysum = 0
            recurse = 0
            for i in self.state_space:
                for newState in i:
                    recurse = max(self.My_Q(newState, self.action_space[0], N-1),self.My_Q(newState, self.action_space[1], N-1),self.My_Q(newState, self.action_space[2], N-1),self.My_Q(newState, self.action_space[3], N-1))
                    mysum += self.p[(state,action,newState)] * recurse
            return self.r[(state,action)] + self.discount_factor * mysum

    def update_Q(self, N):
        for i in self.state_space:
            for state in i:
                for action in self.action_space:
                    self.Q[(state, action)] = self.My_Q(state, action, N)

    def update_optimal_J(self):
        """
        Update J and the optimal policy
        """
        for i in self.state_space:
            for state in i:
                maxi = self.Q[(state, self.action_space[0])]
                u = self.action_space[0]
                for action in self.action_space:
                    if self.Q[(state, action)] > maxi:
                        maxi = self.Q[(state, action)]
                        u = action
                self.optimal_J[state] = maxi
                self.optimal_policy[state] = u

    def update(self):
        self.update_r()
        self.update_p()
        #tune N
        N = 1
        Br = 19
        erreur = 0.5
        e = ((2*(self.discount_factor**N))/(1-self.discount_factor)**2)*Br
        while e > erreur:
            N += 1
            e = ((2*(self.discount_factor**N))/(1-self.discount_factor)**2)*Br
        self.update_Q(N)
        self.update_optimal_J()

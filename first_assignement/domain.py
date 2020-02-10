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

    def reward_signal(self, state, newState):
    	reward = self.g[newState[0]][newState[1]]
    	return reward
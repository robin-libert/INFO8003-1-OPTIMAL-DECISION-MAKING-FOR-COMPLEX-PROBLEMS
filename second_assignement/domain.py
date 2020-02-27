# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:23:06 2020

@author: robin
"""

import math
import random
random.seed(1)

class Domain:

    def __init__(self):
        self.action_space = [-4,4]
        self.m = 1
        self.g = 9.81
        self.discrete_t = 0.1
        self.integration_time_step = 0.001
        self.d_factor = 0.95
        self.p = random.uniform(-0.1,0.1)
        self.s = 0

    def Hill(self, p):
        if p < 0:
            return p**2 + p
        else:
            return p / math.sqrt(1 + (5 * p**2))

    def HillPrime(self, p):
        if p < 0:
            return 2*p + 1
        else:
            return (((1+5*p**2)**(1/2))-((10*(p**2))/(2*(1+5*p**2)**(1/2))))/(1+5*p**2)

    def HillSecond(self, p):
        if p < 0:
            return 2
        else:
            return (-(10*p*(1+5*p**2)**(1/2)-(25*p**3)/((1+5*p**2)^(1/2)))/(1+5*p**2)+(5*p)/((1+5*p**2)**(1/2)))/(1+5*p**2)

    def reward(self, p, s, u):
        pass#neef dynamics
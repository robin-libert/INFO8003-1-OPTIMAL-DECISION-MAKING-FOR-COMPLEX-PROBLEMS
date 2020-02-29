# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:23:06 2020

@author: robin
"""

import math
import random


class Domain:

    def __init__(self):
        self.action_space = [-4, 4]
        self.m = 1
        self.g = 9.81
        self.discrete_t = 0.1
        self.integration_time_step = 0.001
        self.d_factor = 0.95
        self.initial_p = random.uniform(-0.1, 0.1)
        self.initial_s = 0
        self.terminal = False
        self.Br = 1

    def Hill(self, p):
        """
        Function Hill describe in the domain statement
        :param p: the position of the car
        :return:
        """
        if p < 0:
            return p ** 2 + p
        return p / math.sqrt(1 + (5 * p ** 2))

    def HillPrime(self, p):
        """
        Function used to compute the first derivative of the function Hill.
        :param p: the position of the car
        :return: the first derivative of the function Hill
        """
        if p < 0:
            return 2 * p + 1
        return (((1 + 5 * p ** 2) ** (1 / 2)) - ((10 * (p ** 2)) / (2 * (1 + 5 * p ** 2) ** (1 / 2)))) / (
                    1 + 5 * p ** 2)

    def HillSecond(self, p):
        """
        Function used to compute the second derivative of the function Hill.
        :param p: the position of the car
        :return: the second derivative of the function Hill
        """
        if p < 0:
            return 2
        return (-(10 * p * (1 + 5 * p ** 2) ** (1 / 2) - (25 * p ** 3) / ((1 + 5 * p ** 2) ** (1 / 2))) / (
                    1 + 5 * p ** 2) + (5 * p) / ((1 + 5 * p ** 2) ** (1 / 2))) / (1 + 5 * p ** 2)

    def valid_state(self, p, s):
        """
        If the position and the speed of the car are not valid, transform them into valid state.
        :param p: position of the car
        :param s: speed of the car
        :return: a valid position and speed of the car
        """
        pt_copy = p
        st_copy = s
        if pt_copy > 1:
            pt_copy = 1
        elif pt_copy < -1:
            pt_copy = -1
        if st_copy > 3:
            st_copy = 3
        elif st_copy < -3:
            st_copy = -3
        return pt_copy, st_copy

    def dynamics(self, pt, st, ut):
        """
        Compute the next state of the car using the dynamics descibed in the assignement and the euler integration method.
        :param pt: position of the car at time t
        :param st: speed of the car at time t
        :param ut: action taken at time t
        :return: the position and the speed of the car at time t+1
        """
        pt_copy = pt
        st_copy = st
        c = 0
        while c < self.discrete_t:
            s = (ut / (self.m * (1 + self.HillPrime(pt_copy) ** 2))) - (
                        (self.g * self.HillPrime(pt_copy)) / (1 + self.HillPrime(pt_copy) ** 2)) - (
                            ((st_copy ** 2) * self.HillPrime(pt_copy) * self.HillSecond(pt_copy)) / (
                                1 + self.HillPrime(pt_copy) ** 2))
            p = st_copy
            st_copy = st_copy + s * self.integration_time_step
            pt_copy = pt_copy + p * self.integration_time_step
            c += self.integration_time_step
        if abs(pt_copy) > 1 or abs(st_copy) > 3:
            self.terminal = True
        return pt_copy, st_copy

    def reward(self, p, s, u):
        """
        Compute the reward obtained according to a certain state and a certain action
        :param p: position of the car
        :param s: speed of the car
        :param u: action
        :return: reward obtained taking action u from state (p,s)
        """
        newp, news = self.dynamics(p, s, u)
        if newp < -1 or abs(news) > 3:
            return -1
        elif newp > 1 and abs(news) <= 3:
            return 1
        return 0

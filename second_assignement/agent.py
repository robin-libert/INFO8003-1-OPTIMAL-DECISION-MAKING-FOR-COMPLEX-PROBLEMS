# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:59:39 2020

@author: robin
"""
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from display_caronthehill import *


class Agent:

    def __init__(self, domain):
        """
        Initialize the agent which will evolve in a certain domain
        :param domain: the domain
        """
        self.domain = domain
        self.p = random.uniform(-0.1, 0.1)
        self.s = 0
        self.accelerate = domain.action_space[1]
        self.decelerate = domain.action_space[0]

    def move(self, policy):
        """
        Make the agent move according to a policy.
        :param policy:
        :return:
        """
        self.p, self.s = self.domain.dynamics(self.p, self.s, policy)
        self.p, self.s = self.domain.valid_state(self.p, self.s)

    def policy_always_accelerate(self):
        """
        This policy make the agent always accelerate.
        :return: the action of accelerate
        """
        return self.domain.action_space[1]

    def estimate_J(self, policy, n):
        """
        Compute the expected return of a policy according to the Monte Carlo principle.
        :param policy:
        :param n: number of iteration to compute the mean
        :return: estimated expected return of a policy
        """
        r = 0.
        for e in range(n):
            ht = self.run(policy)
            last_reward = ht[-2]
            r += last_reward
        return r / n

    def run(self, policy):
        """
        Make a trajectory given a policy.
        :param policy:
        :return: The trajectory the agent runs.
        """
        self.domain.terminal = False
        self.p = random.uniform(-0.1, 0.1)
        self.s = 0
        ht = []
        ht.append((self.p, self.s))
        while not self.domain.terminal:
            ht.append(policy)
            ht.append(self.domain.reward(self.p, self.s, policy))
            self.move(policy)
            ht.append((self.p, self.s))
        return ht

    def create_video(self, ht):
        fig = plt.figure()

        # Generation of images
        counter = 0
        for e in range(0, len(ht), 3):
            p = ht[e][0]
            s = ht[e][1]
            save_caronthehill_image(p, s, "image\\state" + str(counter) + ".png")
            counter += 1

        # Loading of images
        ims = []
        for i in range(counter):
            image = img.imread("image\\state" + str(i) + ".png")
            im = plt.imshow(image, animated=True)
            ims.append([im])

        # Creation of the animation
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        # ani.save("movie.gif", writer='imagemagick')
        plt.show()

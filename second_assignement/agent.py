# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:59:39 2020

@author: robin
"""
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pickle



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
        self.ACCELERATE_POLICY = 0
        self.RANDOM_POLICY = 1
        self.qn_approximation = []

    def move(self, action):
        """
        Make the agent move according to a policy.
        :param action:
        :return:
        """
        self.p, self.s = self.domain.dynamics(self.p, self.s, action)
        self.p, self.s = self.domain.valid_state(self.p, self.s)

    def policy_always_accelerate(self):
        """
        This policy make the agent always accelerate.
        :return: the action of accelerate
        """
        return self.domain.action_space[1]

    def policy_random(self):
        """
        This policy choose a random action of the action space.
        :return: The chosen action
        """
        return self.domain.action_space[random.randint(0, 1)]

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

    def generate_four_tuples(self, policy, episodes=1000):
        four_tuples_set = []
        for i in range(episodes):
            mytuple = []
            self.domain.terminal = False
            self.p = random.uniform(-0.1, 0.1)
            self.s = 0
            while not self.domain.terminal:
                mytuple.append((self.p, self.s))
                _policy = self.policy_random()
                if policy == self.ACCELERATE_POLICY:
                    _policy = self.policy_always_accelerate()
                elif policy == self.RANDOM_POLICY:
                    _policy = self.policy_random()
                mytuple.append(_policy)
                mytuple.append(self.domain.reward(self.p, self.s, _policy))
                self.move(_policy)
                mytuple.append((self.p, self.s))
                four_tuples_set.append(mytuple)
                mytuple = []
        return four_tuples_set

    def generate_four_tuples2(self, policy, episodes=1000):
        four_tuples_set = []
        for i in range(episodes):
            mytuple = []
            self.domain.terminal = False
            self.p = random.uniform(-0.8, 0.8)
            self.s = random.uniform(-2.5, 2.5)
            while not self.domain.terminal:
                mytuple.append((self.p, self.s))
                _policy = self.policy_random()
                if policy == self.ACCELERATE_POLICY:
                    _policy = self.policy_always_accelerate()
                elif policy == self.RANDOM_POLICY:
                    _policy = self.policy_random()
                mytuple.append(_policy)
                mytuple.append(self.domain.reward(self.p, self.s, _policy))
                self.move(_policy)
                mytuple.append((self.p, self.s))
                four_tuples_set.append(mytuple)
                mytuple = []
        return four_tuples_set

    def fitted_q_iteration(self, four_tuples_set, stop, algo="Linear Regression"):
        N = 0
        print("Computing fitted q iteration ...")
        print("{} / 100".format((N/stop)*100))
        while N < stop:  # temp stopping condition
            N += 1
            X = []
            y = []
            for t in four_tuples_set:
                X.append([t[0][0], t[0][1], t[1]])
                if N == 1:
                    y.append(t[2])
                else:
                    y.append(t[2] + self.domain.d_factor *
                             max(self.qn_approximation[-1].predict(np.array([t[3][0], t[3][1], -4]).reshape(1, -1))[0],
                                 self.qn_approximation[-1].predict(np.array([t[3][0], t[3][1], 4]).reshape(1, -1))[0]))
            if algo == "Linear Regression":
                model = LinearRegression()
            elif algo == "Extremely Randomized Trees":
                model = ExtraTreesRegressor(n_estimators=10)
            elif algo == "Neural Network":
                X = np.array(X)
                y = np.array(y)
                model = Sequential()
                model.add(Dense(9, input_dim=3, activation='relu'))
                model.add(Dense(9, activation='relu'))
                model.add(Dense(9, activation='relu'))
                model.add(Dense(9, activation='relu'))
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
                # model.fit(X, y, epochs=10, batch_size=10, verbose=0)
                model.fit(X, y, epochs=10, batch_size=10, verbose=0)
                self.qn_approximation.append(model)
            if algo != "Neural Network":
                self.qn_approximation.append(model.fit(np.array(X), np.array(y)))
            print("{} / 100".format((N / stop) * 100))
        return self.qn_approximation[-1]

    def display_q_function(self, four_tuples_set, stop, algo="Linear Regression"):
        """
        method to display the estimator of the q function for the action 'forward', 'backward' and the max of
        the 'forward' and 'backward'
        :param n: index if the Qn to display
        :param method: string to choose the learning method (Linear_regression, Extremely_Randomized_Trees
                                                                                                or Neural_network)
        :return: plot three images
        """
        definition = 50
        x = np.linspace(-1, 1, definition)
        y = np.linspace(-3, 3, definition)
        qn = self.fitted_q_iteration(four_tuples_set, stop, algo)
        vector_img_right = np.zeros((definition, definition))
        vector_img_left = np.zeros((definition, definition))
        vector_img = np.zeros((definition, definition))
        print("Predict estimation of Q...")
        print("{} / 100".format(0))
        for i in range(definition):
            for j in range(definition):
                vector_img_right[j, i] = qn.predict(np.array([x[i], y[j], 4]).reshape(1, -1))[0]
                vector_img_left[j, i] = qn.predict(np.array([x[i], y[j], -4]).reshape(1, -1))[0]
                if vector_img_left[j, i] <= vector_img_right[j, i]:
                    vector_img[j, i] = 4
                else:
                    vector_img[j, i] = -4
            print("{} / 100".format((i / definition)*100))
        print(vector_img)
        cs = plt.contourf(x, y, vector_img_right, cmap='Spectral')
        plt.colorbar(cs)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("Value of Q" + str(stop) + " for each state and the forward action")
        plt.show()
        plt.clf()

        cs2 = plt.contourf(x, y, vector_img_left, cmap='Spectral')
        plt.colorbar(cs2)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("Value of Q" + str(stop) + " for each state and the backward action")
        plt.show()
        plt.clf()

        cs3 = plt.contourf(x, y, vector_img, cmap='Spectral')
        plt.colorbar(cs3)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("argmax of Q" + str(stop))
        plt.show()


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
            _policy = self.policy_random()
            if policy == self.ACCELERATE_POLICY:
                _policy = self.policy_always_accelerate()
            elif policy == self.RANDOM_POLICY:
                _policy = self.policy_random()
            ht.append(_policy)
            ht.append(self.domain.reward(self.p, self.s, _policy))
            self.move(_policy)
            ht.append((self.p, self.s))
        return ht

    def run_optimal_policy(self, model_name):
        model = pickle.load(open(model_name, 'rb'))
        self.domain.terminal = False
        self.p = random.uniform(-0.1, 0.1)
        self.s = 0
        ht = []
        ht.append((self.p, self.s))
        print(model.predict(np.array([self.p, self.s, -4]).reshape(1, -1))[0])
        """i = 1
        while not self.domain.terminal:
            if model.predict(np.array([self.p, self.s, -4]).reshape(1, -1))[0] <= model.predict(np.array([self.p, self.s, 4]).reshape(1, -1))[0]:
                _policy = 4
            else:
                _policy = -4
            ht.append(_policy)
            ht.append(self.domain.reward(self.p, self.s, _policy))
            self.move(_policy)
            ht.append((self.p, self.s))
            print(i)
            i+=1
        print(ht)
        print(len(ht))
        return ht"""


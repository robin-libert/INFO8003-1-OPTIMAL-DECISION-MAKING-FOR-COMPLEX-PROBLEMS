# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:59:22 2020

@author: robin
"""

from domain import Domain
import random

def tune_N(gamma=0.99, Br=19, erreur = 0.5):
    """
    return the best value of N
    """
    N = 1
    e = ((2*(gamma**N))/(1-gamma)**2)*Br
    while e > erreur:
        N += 1
        e = ((2*(gamma**N))/(1-gamma)**2)*Br
    return N

def create_trajectory(size):
    ht = []
    x = random.randint(0,4)
    y = random.randint(0,4)
    initialState = domain.state_space[y][x]
    ht.append(initialState)
    for i in range(size):
        action = domain.action_space[random.randint(0,3)]
        arrivalState = domain.move(initialState, action)
        reward = domain.reward_signal(arrivalState)
        ht.append(action)
        ht.append(reward)
        ht.append(arrivalState)
        initialState = arrivalState
    return ht

def init_Q():
    Q = {}
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                Q[(state, action)] = 0
    return Q

def compute_Q(trajectory, Q_dict , alpha=0.05):
    t = ((len(trajectory)-1)/3)
    k=0
    while k < t:
        k_index = k
        xk = trajectory[k_index * 3]
        uk = trajectory[k_index * 3 + 1]
        rk = trajectory[k_index * 3 + 2]
        xkplus1 = trajectory[k_index * 3 + 3]
        Q_dict[(xk,uk)] = (1-alpha) * Q_dict[(xk,uk)]
        maxi = 0
        for u in domain.action_space:
            temp = Q_dict[(xkplus1, u)]
            if temp > maxi:
                maxi = temp
        Q_dict[(xk,uk)] += alpha * (rk + domain.discount_factor * maxi)
        k += 1

setting = int(input("Press 0 for deterministic setting or 1 for stochastic setting"))
n = int(input("Choose how many trajectories you want to generate"))
length = int(input("Choose the length of the trajectories you want to generate"))
print("wait until completion")
domain = Domain()
domain.setting = setting
Q = init_Q()
for i in range(n):
    trajectory = create_trajectory(length)
    compute_Q(trajectory, Q)

#display JN according to an optimal policy
for i in domain.state_space:
    for state in i:
        maxi = Q[(state, domain.action_space[0])]
        u = domain.action_space[0]
        for action in domain.action_space:
            if Q[(state, action)] > maxi:
                maxi = Q[(state, action)]
                u = action
        print("{} Estimation J* = {} --- Real J* = {}".format(state, maxi, domain.optimal_J[state]))
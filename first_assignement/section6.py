# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:59:22 2020

@author: robin
"""

from domain import Domain
import random
random.seed(42)

domain = Domain()

def my_routine(T):
    """
    Compute r(x,u) and p(x'|x,u) and a trajectory of size T
    """
    p = {}
    r = {}
    #counters to compute the mean
    nr = {}
    np = {}
    counter = 0
    ht = []
    for i in domain.state_space:
        for state in i:
            for j in domain.state_space:
                for newState in j:
                    for action in domain.action_space:
                        r[(state, action)] = 0.
                        p[(state, action, newState)] = 0.
                        nr[(state, action)] = 0.
                        np[(state, action, newState)] = 0.
    state = (3,0)#initial state
    ht.append(state)
    while counter < T:
        action = domain.action_space[random.randint(0,3)]
        ht.append(action)
        newState = domain.move(state, action) #new state with maybe some disturbance
        reward = domain.reward_signal(newState)
        ht.append(reward)
        ht.append(newState)
        newState2 = (min(max(state[0]+action[0],0),domain.n-1), min(max(state[1]+action[1],0),domain.m-1))#new state if there is no disturbances
        r[(state, action)] += reward
        nr[(state, action)] += 1.
        if newState == newState2:
            p[(state, action, newState2)] += 1.
            np[(state, action, newState2)] += 1.
        else:
            np[(state, action, newState2)] += 1.
        state = newState
        counter += 1
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                if nr[(state, action)] > 0:
                    r[(state, action)] = r[(state, action)] / nr[(state, action)]
    for i in domain.state_space:
        for state in i:
            for j in domain.state_space:
                for newState in j:
                    for action in domain.action_space:
                        if np[(state, action, newState)] != 0:
                            p[(state, action, newState)] = p[(state, action, newState)] /np[(state, action, newState)]
    return (p, r, ht)



def memoize(f):
    """
    To optimize the Q function
    """
    memo = {}
    def helper(a,b,c,d,e):
        if (a,b,c) not in memo:
            memo[(a,b,c)] = f(a,b,c,d,e)
        return memo[(a,b,c)]
    return helper

@memoize
def Q(state, action, N, r, p):
    """
    Return the value of the state_action value function.
    state:
        a tuple (n,m) where 0<=n,m<=4
    action:
        a tuple (a,b) where -1<=a,b<= 1 which belongs to domain.action_space
    N:
        the nnumber of steps
    r:
        dictionary containing r(x,u) previously computed for each state, action
    p:
        dictionary containing p(x'|x,u) previously computed for each state, action, new state
    """
    if N == 0:
        return 0
    else:
        mysum = 0
        recurse = 0
        for i in domain.state_space:
            for newState in i:
                recurse = max(Q(newState, domain.action_space[0], N-1, r, p),Q(newState, domain.action_space[1], N-1, r, p),Q(newState, domain.action_space[2], N-1, r, p),Q(newState, domain.action_space[3], N-1, r, p))
                mysum += p[(state,action,newState)] * recurse
        return r[(state,action)] + domain.discount_factor * mysum

def compute_JN_and_optimal_policy(N, rewards, probabilities):
    """
    Compute an optimal policy and the value function
    """
    optimal_J_mu_N = {}
    optimal_policy = {}
    for i in domain.state_space:
        for state in i:
            best_action = None
            maxi = 0
            for action in domain.action_space:
                current = Q(state, action,N, rewards, probabilities)
                if current > maxi:
                    maxi = current
                    best_action = action
            optimal_policy[state] = best_action
            optimal_J_mu_N[state] = maxi
    return (optimal_J_mu_N, optimal_policy)

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

def compute_Q(state, action, trajectory, Q_dict, k=0, alpha=0.05):
    t = ((len(trajectory)-1)/3)
    while k < t:
        k_index = k
        xk = trajectory[k_index * 3]
        uk = trajectory[k_index * 3 + 1]
        rk = trajectory[k_index * 3 + 2]
        xkplus1 = trajectory[k_index * 3 + 3]
        Q_dict[(xk,uk)] = (1-alpha) * Q_dict[(xk,uk)]
        maxi = 0
        for u in domain.action_space:
            temp = Q_dict[(xkplus1, action)]
            if temp > maxi:
                maxi = temp
        Q_dict[(xk,uk)] += alpha * (rk + domain.discount_factor * maxi)
        k += 1
    return Q_dict[(state, action)]

trajectory = create_trajectory(10000000)
Q_dict = init_Q()
compute_Q((0,0), (0,1), trajectory, Q_dict)

optimal_policy = {}
J = {}

for i in domain.state_space:
    for state in i:
        maxi = 0
        u = None
        for action in domain.action_space:
            if Q_dict[(state, action)] > maxi:
                maxi = Q_dict[(state, action)]
                u = action
        optimal_policy[state] = u
        J[state] = maxi

T = 10000
N = tune_N()#compute N to have an error <= 0.5
probabilities, rewards, trajectory2 = my_routine(T)

optimal_J_mu_N, optimal_policy2 = compute_JN_and_optimal_policy(N, rewards, probabilities)
for i in domain.state_space:
    for state in i:
        print('current state = {} | optimal_JN = {} | estimate_JN = {}'.format(state, optimal_J_mu_N[state], J[state]))
print(optimal_policy)
print(optimal_policy2)

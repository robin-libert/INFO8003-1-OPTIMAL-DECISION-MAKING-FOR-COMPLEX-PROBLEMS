from domain import Domain
import random

domain = Domain()

def display_random_policy(state, N=100, g=domain.g):
    """
    Display the trajectory of our random policy in the console.
    state : a tuple (x,y)
            the state where the bot start
    N : integer
        the number of steps (100 by default)
    g : 2x2 matrix containing the rewards
        by default, rewards (the matrix of rewards given in the assignement)
    """
    t = 0
    while t <= N:
        action = domain.action_space[random.randint(0,3)]
        newState = domain.move(state, action)
        reward = domain.reward_signal(state, newState)
        print('(x_{0} = {1}, u_{0} = {2}, r_{0} = {3}, x_{4} = {5})'.format(t,state, action, reward, t+1, newState))
        t+=1
        state = newState

print('First, choose the coordonates where the bot start.')
x , y = (int(input('Choose x between 0 and {} : '.format(domain.n))), int(input('Choose y between 0 and {}: '.format(domain.m))))
N = int(input("Choose an integer which will be the number of steps of the bot : "))
display_random_policy((x,y), N)

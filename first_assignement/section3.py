from domain import Domain
import random

domain = Domain()

def expected_return_random_policy(state, N=999, g=domain.g, gamma=domain.discount_factor):
    if N == 0:
        return 0
    else:
        action = domain.action_space[random.randint(0,3)]
        newState = domain.move(state, action)
        return domain.reward_signal(state, newState) + gamma * expected_return_random_policy(newState, N-1)

def display_expected_return_random_policy(N = 999, g=domain.g):
    for i in domain.state_space:
        for j in i:
            print('x_0 = {} expected return = {}'.format(j,expected_return_random_policy(j,N)))

display_expected_return_random_policy(N=2000)
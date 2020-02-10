from domain import Domain

domain = Domain()

def tune_N(gamma=domain.discount_factor, Br=19, erreur = 0.05):
    N = 1
    e = ((gamma**N)/(1-gamma))*Br
    while e > erreur:
        N += 1
        e = ((gamma**N)/(1-gamma))*Br
    return N


def expected_return_right_policy(state, N=999, g=domain.g, gamma=domain.discount_factor):
    if N == 0:
        return 0
    else:
        action = domain.action_space[2]
        newState = domain.move(state, action)
        return domain.reward_signal(state, newState) + gamma * expected_return_right_policy(newState, N-1)

def display_expected_return_right_policy(N = 999, g=domain.g):
    for i in domain.state_space:
        for j in i:
            print('x_0 = {} expected return = {}'.format(j,expected_return_right_policy(j,N)))


n = tune_N()
print("Deterministic domain")
print("--------------------")
domain.setting = 0
display_expected_return_right_policy(N=2000)
print("Stochastic domain")
print("--------------------")
domain.setting = 1
display_expected_return_right_policy(N=2000)
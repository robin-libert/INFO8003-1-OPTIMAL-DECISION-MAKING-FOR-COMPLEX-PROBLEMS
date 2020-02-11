from domain import Domain

domain = Domain()

down_policy = 0
up_policy = 1
right_policy = 2
left_policy = 3

def tune_N(gamma=domain.discount_factor, Br=19, erreur = 0.05):
    N = 1
    e = ((gamma**N)/(1-gamma))*Br
    while e > erreur:
        N += 1
        e = ((gamma**N)/(1-gamma))*Br
    return N


def expected_return(state, policy = right_policy, N=999, g=domain.g, gamma=domain.discount_factor):
    if N == 0:
        return 0
    else:
        action = domain.action_space[policy]
        newState = domain.move(state, action)
        return domain.reward_signal(newState) + gamma * expected_return(newState, policy=policy, N=N-1)

def display_expected_return(N = 999, policy = right_policy, g=domain.g):
    for i in domain.state_space:
        for j in i:
            print('x_0 = {} expected return = {}'.format(j,expected_return(j,N=N, policy = policy)))


n = tune_N()
print("Deterministic domain")
print("--------------------")
domain.setting = 0
display_expected_return(N=2000, policy=right_policy)
print("Stochastic domain")
print("--------------------")
domain.setting = 1
display_expected_return(N=2000, policy=right_policy)
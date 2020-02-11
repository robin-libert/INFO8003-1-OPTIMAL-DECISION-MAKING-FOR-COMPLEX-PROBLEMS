from domain import Domain
import random
random.seed(42)#for reproductibility

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
            best_action = domain.action_space[0]
            maxi = Q(state, domain.action_space[0],N, rewards, probabilities)
            for action in domain.action_space[1:4]:
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


domain.setting = 0
T = 1000000
N = tune_N()#compute N to have an error <= 0.5
probabilities, rewards, trajectory = my_routine(T)

optimal_J_mu_N, optimal_policy = compute_JN_and_optimal_policy(N, rewards, probabilities)
for i in domain.state_space:
    for state in i:
        print('current state = {} | optimal_JN = {}'.format(state, optimal_J_mu_N[state]))



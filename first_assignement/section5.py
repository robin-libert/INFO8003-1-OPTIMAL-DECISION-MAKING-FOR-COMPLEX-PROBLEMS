from domain import Domain
import random
random.seed(42)

domain = Domain()

#section 5 -----------------------------------------------------------------------
Q = {}
p = {}
r = {}

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

def estimate_r(state, action, trajectory):
    A = []
    t = 0
    sum_rewards = 0
    while t < len(trajectory)-3:
        currentState = trajectory[t]
        currentAction = trajectory[t+1]
        currentReward = trajectory[t+2]
        if currentState == state and currentAction == action:
            A.append(t)
            sum_rewards += currentReward
        t += 3
    if len(A)>0:
        return sum_rewards / len(A)
    else:
        return 0

def estimate_p(state, action, newState, trajectory):
    A = []
    t = 0
    sum_p = 0
    while t < len(trajectory)-3:
        currentState = trajectory[t]
        currentAction = trajectory[t+1]
        currentNewState = trajectory[t+3]
        if currentState == state and currentAction == action:
            A.append(t)
            if currentNewState == newState:
                sum_p += 1
        t += 3
    if len(A)>0:
        return sum_p / len(A)
    else:
        return 0

def estimate_all_r(trajectory):
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                r[(state, action)] = estimate_r(state, action, trajectory)
    return r

def estimate_all_p(trajectory):
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                for j in domain.state_space:
                    for newState in j:
                        p[(state, action, newState)] = estimate_p(state, action, newState, trajectory)
    return p

def memoize(f):
    """
    To optimize the Q function
    """
    memo = {}
    def helper(a,b,c):
        if (a,b,c) not in memo:
            memo[(a,b,c)] = f(a,b,c)
        return memo[(a,b,c)]
    return helper

@memoize
def estimate_Q(state, action, N):
    if N == 0:
        return 0
    else:
        mysum = 0
        recurse = 0
        for i in domain.state_space:
            for newState in i:
                recurse = max(estimate_Q(newState, domain.action_space[0], N-1),estimate_Q(newState, domain.action_space[1], N-1),estimate_Q(newState, domain.action_space[2], N-1),estimate_Q(newState, domain.action_space[3], N-1))
                mysum += p[(state,action,newState)] * recurse
        return r[(state,action)] + domain.discount_factor * mysum

domain.setting = int(input("Press 0 for deterministic setting or 1 for stochastic setting"))
length = int(input("Choose the length of the trajectory you want to generate"))
domain.update()
trajectory = create_trajectory(length)
estimate_all_r(trajectory)
estimate_all_p(trajectory)

for i in domain.state_space:
    for state in i:
        for action in domain.action_space:
            Q[(state, action)] = estimate_Q(state, action,1000)

for i in domain.state_space:
    for state in i:
        maxi = Q[(state, domain.action_space[0])]
        u = domain.action_space[0]
        for action in domain.action_space:
            if Q[(state, action)] > maxi:
                maxi = Q[(state, action)]
                u = action
        print("{} Estimation J* = {} --- Real J* = {}".format(state, maxi, domain.optimal_J[state]))
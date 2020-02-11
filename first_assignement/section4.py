from domain import Domain
import random

domain = Domain()

#section 4 -----------------------------------------------------------------------
def r(T):
    """
    Compute r(x,u) for each state, using a uniform random policy
    T: integer
        the greater T is, the more acurate is the prediction
    return a dictionary containing each r(x,u)
    """
    d = {}
    n= {}
    #init the dictionary
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                d[(state, action)] = 0
                n[(state, action)] = 0
    counter = 0
    while counter < T:
        for i in domain.state_space:
            for state in i:
                action = domain.action_space[random.randint(0,3)]
                newState = domain.move(state, action)
                d[(state, action)] += domain.reward_signal(state, newState)
                n[(state, action)] += 1
        counter += 1
    for i in domain.state_space:
        for state in i:
            for action in domain.action_space:
                if n[(state, action)] != 0:
                    d[(state, action)] = d[(state, action)] / n[(state, action)]

    return d

def p(T):
    """
    Compute P(x'|x,u) for each state and each action, using a uniform random policy
    T: integer
        the greater T is, the more acurate is the prediction
    return a dictionary containing each P(x'|x,u)
    """
    d = {}
    n= {}
    #init the dictionary
    for i in domain.state_space:
        for state in i:
            for j in domain.state_space:
                for newState in j:
                    for action in domain.action_space:
                        d[(state, action, newState)] = 0
                        n[(state, action, newState)] = 0
    counter = 0
    while counter < T:
        for i in domain.state_space:
            for state in i:
                for j in domain.state_space:
                    for newState in j:
                        action = domain.action_space[random.randint(0,3)]
                        newState2 = domain.move(state, action)
                        if newState == newState2:
                            d[(state, action, newState)] += 1
                            n[(state, action, newState)] += 1
                        else:
                            n[(state, action, newState)] += 1
        counter += 1
    for i in domain.state_space:
        for state in i:
            for j in domain.state_space:
                for newState in j:
                    for action in domain.action_space:
                        if n[(state, action, newState)] != 0:
                            d[(state, action, newState)] = d[(state, action, newState)] /n[(state, action, newState)]
    return d

def memoize(f):
    memo = {}
    def helper(a,b,c,d,e):
        if (a,b,c) not in memo:
            memo[(a,b,c)] = f(a,b,c,d,e)
        return memo[(a,b,c)]
    return helper

@memoize
def Q(state, action, N, r, p):
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



T = 100
domain.setting = 0
rewards = r(T)
probabilities = p(T)
for action in domain.action_space:
    print(Q((3,0), action,2, rewards, probabilities))




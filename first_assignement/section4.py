from domain import Domain
import random

domain = Domain()

#section 4 -----------------------------------------------------------------------
def probabilities(state, N=999):
    p = []
    n = N
    for i in range(len(state_space)):
        p.append([])
        for j in range(len(state_space[i])):
            p[i].append(0)
    while N >= 0:
        for i in state_space:
            for j in i:
                if dynamics(state, action_space[random.randint(0,3)]) == j:
                    p[j[1]][j[0]] += 1
        N -= 1
    for i in range(len(p)):
        for j in range(len(p[i])):
            p[i][j] = p[i][j]/n
    return p

def all_probabilities(N):
    p = []
    for i in state_space:
        for j in i:
            p.append(probabilities(j,N))
    return np.array(p)
print(all_probabilities(N=7000))



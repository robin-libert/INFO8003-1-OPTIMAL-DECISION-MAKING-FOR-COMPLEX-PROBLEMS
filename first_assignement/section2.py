from domain import Domain
domain = Domain()

def display_right_policy(state, N=100, g=domain.g):
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
        action = domain.action_space[2]
        newState = domain.move(state, action)
        reward = domain.reward_signal(newState)
        print('(x_{0} = {1}, u_{0} = {2}, r_{0} = {3}, x_{4} = {5})'.format(t,state, action, reward, t+1, newState))
        t+=1
        state = newState

N = int(input("Choose an integer which will be the number of steps of the bot : "))
print("Deterministic domain")
print("--------------------")
domain.setting = 0
display_right_policy((3,0), N)
print("Stochastic domain")
print("--------------------")
domain.setting = 1
display_right_policy((3,0), N)

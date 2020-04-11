from domain import Domain
from agent import Agent
import random
import pickle
import numpy as np


def stop_condition_1(domain, error=0.05):
    N = 0
    cond = (2 * domain.d_factor ** N * domain.Br) / (1 - domain.d_factor) ** 2
    while cond > error:
        N += 1
        cond = (2 * domain.d_factor ** N * domain.Br) / (1 - domain.d_factor) ** 2
    return N


if __name__ == "__main__":
    random.seed(2)
    domain = Domain()
    agent = Agent(domain)
    policy = agent.RANDOM_POLICY
    print("Generating the set of four tuples ...")
    four_tuples_set = agent.generate_four_tuples(policy)
    #four_tuples_set = agent.generate_four_tuples2(policy)
    print("Generation of the set completed.")
    print("Compute statistics over the set...")
    positiveRewards = 0
    negativeRewards = 0
    for t in four_tuples_set:
        if t[2] == 1:
            positiveRewards += 1
        elif t[2] == -1:
            negativeRewards += 1
    # print(four_tuples_set)
    print("Length of the set is {}".format(len(four_tuples_set)))
    print("Number of negatives rewards is {}".format(negativeRewards))
    print("Number of positives rewards is {}".format(positiveRewards))

    stop = stop_condition_1(domain)
    # agent.display_q_function(four_tuples_set, 50, algo="Linear Regression")
    agent.display_q_function(four_tuples_set, 20, algo="Neural Network")
    # agent.display_q_function(four_tuples_set, stop, algo="Extremely Randomized Trees")
    pickle.dump(agent.qn_approximation[-1], open("20_NN_1", 'wb'))
    # model = pickle.load(open("50_linear_regression", 'rb'))
    #agent.run_optimal_policy("50_LIN_1")
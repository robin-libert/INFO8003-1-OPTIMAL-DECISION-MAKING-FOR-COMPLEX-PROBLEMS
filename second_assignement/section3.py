from domain import Domain
from agent import Agent

if __name__ == "__main__":
    domain = Domain()
    agent = Agent(domain)
    policy = agent.RANDOM_POLICY
    # Compute the expected return of the policy above over 100 iterations (could be slow)
    J_estimation = agent.estimate_J(policy, 100)
    print("Expected return of the policy always accelerate: {}".format(J_estimation))

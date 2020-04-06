from domain import Domain
from agent import Agent
from utils import *

if __name__ == "__main__":
    domain = Domain()
    agent = Agent(domain)
    policy = agent.ACCELERATE_POLICY
    ht = agent.run(policy)
    print("Creating a gif called trajectory.gif in the current directory.")
    print("Please wait ...")
    save_GIF(ht)
    print("Creation completed.")
import gymnasium as gym
import pygame
import time
import random
from collections import deque


num_episodes = 1000
num_steps = 100

epsilon = 1.0

seed = 42
env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="ansi", is_slippery=False)


# initialize the quantum circuit
##TBD

# initialize the replay buffer
##TBD
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        pass

    def sample(self, batch_size):
        pass

def adjusted_reward(reward, done):
    if reward == 0:
        if done:
            return -1.0
        else:
            return -0.01
    return reward

def qc(s_t, theta)
    return E

def a_maker(E):
    return np.argmax(E)



# initialize the replay buffer
buffer = ReplayBuffer(capacity=10000)

for episode in range(num_episodes):
    state, info = env.reset(seed=seed)
    done = False
    for step in range(num_steps):
        # sample a number between 0 and 1
        sample_number = np_random.random()
        if sample_number < epsilon:
            action = env.action_space.sample()
        else:
            # use quantum circuit to determine the action
            # action = ??
            pass
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward = adjusted_reward(reward, done)

        # store the experience in the replay buffer
        transition = (state, action, reward, next_state, done)
        buffer.push(transition)
        state = next_state

        # update the quantum circuit
        transitions = buffer.sample(batch_size=32)

        # update quantum circuit with transitions
        ## compute TD target
        ## compute loss
        ## update quantum circuit parameters
        ## update quantum circuit parameters of the target network every C steps


        if done:
            break
        pass

import torch
import gymnasium as gym
import numpy.random as np_random
from collections import deque


num_episodes = 1000
num_steps = 100

epsilon = 1.0

seed = 42
env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="ansi", is_slippery=False)



# initialize the replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Minimal Experience Replay Buffer
        (state, action, reward, next_state, done)

        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Store a single transition in the replay buffer
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of stored transitions from the replay buffer
        """
        return random.sample(self.buffer, batch_size)

    def length(self):
        """
        Returns the current number of stored transitions in the buffer
        """
        return len(self.buffer)

def adjusted_reward(reward, done):
    if reward == 0:
        if done:
            return -1.0
        else:
            return -0.01
    return reward

# Compute TD target
def compute_td_target(transitions, gamma=0.99):
    td_targets = []
    for state, action, reward, next_state, done in transitions:
        next_state_tensor = decimalToBinaryFixLength(4, next_state)
        next_action = get_action_from_quantum_circuit(next_state_tensor, env.action_space.n)
        td_target = reward + (0 if done else gamma * next_action)
        td_targets.append(td_target)
    return torch.tensor(td_targets)

# Compute loss
def compute_loss(q_values, td_targets):
    return torch.nn.functional.mse_loss(q_values, td_targets)

def loss_func(q, y):
  


def qc(st, theta):
  return E_list

def a_maker(E_list):
    return E_list.index(max(E_list))

def deep_Q_Learning():
  buffer_capacity = 32
  buffer = ReplayBuffer(capacity=buffer_capacity)
  env.reset(seed=seed)

  for episode in range(num_episodes):
      env.reset()
      done = False
      theta  = (torch.rand(n) - 0.5) * torch.pi #
      theta_ = (torch.rand(n) - 0.5) * torch.pi # target net

      for step in range(num_steps):
          # sample a number between 0 and 1
          sample_number = np_random.random()
          if sample_number < epsilon:
              action = env.action_space.sample()
          else:
              action = a_maker(qc(st, theta))

          next_state, reward, terminated, truncated, info = env.step(action)
          done = terminated or truncated
          reward = adjusted_reward(reward, done)

          # store the experience in the replay buffer
          transition = (state, action, reward, next_state, done)
          buffer.push(transition)
          state = next_state

          # update the quantum circuit
          if buffer.legth() < buffer_capacity + 1:
            break

          transitions = buffer.sample(batch_size=10)


          # update quantum circuit with transitions
          ## compute TD target
          td_targets = compute_td_target(transitions, gamma=0.99)

          ## compute loss
          loss = compute_loss(q_values, td_targets)


          ## update quantum circuit parameters
          ## update quantum circuit parameters of the target network every C steps


          if done:
              break
          pass
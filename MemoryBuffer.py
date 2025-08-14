import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Minimal Experience Replay Buffer
        (state, action, reward, next_state, done)

        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Store a single transition in the replay buffer
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of stored transitions from the replay buffer
        """
        return random.sample(self.buffer, batch_size)

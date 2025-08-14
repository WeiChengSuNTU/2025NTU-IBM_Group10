import gymnasium as gym
import numpy.random as np_random
from qiskit import QuantumCircuit
from qiskit import transpile
import numpy as np
from qiskit_aer import Aer
import torch
import torch.nn as nn

num_episodes = 1000
num_steps = 100

epsilon = 1.0

seed = 42
env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="ansi", is_slippery=False)

gemma = 0.99

from collections import deque
import random

# initialize the quantum circuit
# Hank
#================ qc setting ==================================
n_qubits = 4    # for 16 states
n_layers = 2
n_params = n_qubits * n_layers * 2  # 2 parameters per qubits
qc = QuantumCircuit(n_qubits)

#================ state trans =================================
# decimal -> binary
def dec_to_bin(state, n):
    return format(state, f'0{n}b')

#================ create qc ===================================
# encoding
def encoding_layer(state):
    for i in range(n_qubits):
        if dec_to_bin(state, n_qubits)[i] == '1':
            qc.rx(np.pi, i)  # apply RX gate with pi rotation
        else:
            qc.rx(0, i)  # apply RX gate with 0 rotation

# entanglement using CNOT gates
def entanglement_layer():
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

# Apply parameterized gates
def variational_layer(weights):
    for i in range(n_qubits):
        qc.ry(weights[i], i)
        qc.rz(weights[i], i)

# ==============================================================
def Q_value_list(iuput_state, weights, shots=1024):
    encoding_layer(iuput_state)

    for _ in range(n_layers):
        entanglement_layer()
        qc.barrier()
        variational_layer(weights)
        qc.barrier()

    qc.measure_all()

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(transpile(qc, backend), shots=shots)
    result = job.result()
    counts = result.get_counts()

    expectation_list = []
    for qubit in range(n_qubits):
        prob = 0
        for bitstring, count in counts.items():
            if bitstring[::-1][qubit] == '1':
                prob += count
        expectation = prob / shots
        expectation_list.append(expectation)

    return expectation_list


class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.randn(n_qubits * n_layers * 2))  # 2 parameters per qubit

    def forward(self, x):
        q_out = Q_value_list(x, self.weights.detach().numpy(), shots=1024)
        return torch.tensor(q_out, dtype=torch.float32)


# initialize the replay buffer
# Adam
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
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of stored transitions from the replay buffer
        """
        return random.sample(self.buffer, batch_size)



def adjusted_reward(reward, done):
    if reward == 0:
        if done:
            return -1.0
        else:
            return -0.01
    return reward


buffer = ReplayBuffer(capacity=80)

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
            q_values = qc
            action = int(np.argmax(q_values))
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
        td_targets = reward + gemma * (1 - dones) * torch.max(transition, dim=1)[0]
        ## compute loss
        loss = torch.nn.MSELoss(q_values.gather(1, actions.unsqueeze(1)).squeeze(), td_targets)
        ## update quantum circuit parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## update quantum circuit parameters of the target network every C steps


        if done:
            break
        else:
            continue
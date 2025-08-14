from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit_machine_learning.connectors import TorchConnector

import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import numpy.random as np_random
from collections import deque

num_episodes = 1000
num_steps = 100

seed = 42
env_name = "FrozenLake-v1"
#env = gym.make(env_name, render_mode="ansi", is_slippery=False)
env = gym.make(env_name, render_mode="human", is_slippery=False)

state_space_size = env.observation_space.n  # FrozenLake 是 16
n_qubits = math.ceil(math.log2(state_space_size))  # → 4 qubits

gemma = 0.99

from collections import deque
import random

# initialize the quantum circuit
# Hank
#================ qc setting ==================================
n_qubits = 4    # for 16 states
n_layers = 1
n_params = n_qubits * n_layers * 2  # 2 parameters per qubits
shots = 1024
qc = QuantumCircuit(n_qubits)

#================ state trans =================================
# decimal -> binary
# def dec_to_bin(state, n):
    # if isinstance(state, torch.Tensor):
        # if state.dim() > 0 and state.numel() > 1:
            # state = int(torch.argmax(state).item())  # one-hot → index
        # else:
            # state = int(state.item())  # scalar tensor
    # return format(state, f'0{n}b')

def dec_to_bin(state, n_qubits):
    if isinstance(state, torch.Tensor):
        if state.numel() > 1:
            state = int(torch.argmax(state).item())  # one-hot → index
        else:
            state = int(state.item())
    return format(state, f'0{n_qubits}b')


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
        qc.rz(weights[i + n_qubits], i)

# ==============================================================
def Q_value_list(input_state, weights, shots=shots):
    encoding_layer(input_state)

    for _ in range(n_layers):
        qc.barrier()
        entanglement_layer()
        qc.barrier()
        variational_layer(weights)

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

    return torch.tensor(expectation_list, dtype=torch.float32)


class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_weights = n_qubits * n_layers * 2
        self.weights = nn.Parameter(torch.randn(n_qubits * n_layers * 2))  # 2 parameters per qubit
        self.sparse = False
        self.num_inputs = n_qubits

    def forward(self, x):
        q_out = Q_value_list(x, self.weights.detach().numpy(), shots=1024)
        # return torch.tensor(q_out, dtype=torch.float32)
        return q_out.clone().detach().float()



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
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

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




def deep_Q_Learning():
    #epsilon = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    buffer_capacity = 32
    buffer = ReplayBuffer(capacity=80)
    vqc = VQC(n_qubits, n_layers)  # Initialize the quantum circuit model
    quantum_model_torch = vqc  # 將量子電路模型連接到 PyTorch
    optimizer = torch.optim.Adam(quantum_model_torch.parameters(), lr=0.001)  # 使用 Adam 優化器

    env.reset(seed=seed)
    for episode in range(1, num_episodes + 1):
        state, info = env.reset(seed=seed)
        done = False

        # 確保 state 是張量形式
        # state = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor

        for step in range(num_steps):
            # sample a number between 0 and 1
            sample_number = np_random.random()
            if sample_number < epsilon:
                action = env.action_space.sample()  # 隨機選擇行動
                print(f"[Exploration] ε={epsilon:.3f}, action={action}")
            else:
                # 使用量子電路計算 Q 值
                # state_tensor = torch.nn.functional.one_hot(
                    # torch.tensor(state), num_classes=state_space_size
                # ).float()
                # q_values = quantum_model_torch(state_tensor)  # 確保這裡計算 q_values
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                if not done:
                    q_values = q_network(state_tensor)
                    print("q_values shape:", q_values.shape)
                    print("actions:", actions)
                    q_values_selected = torch.gather(q_values, 1, torch.tensor([[actions]]))
                else:
                    q_values_selected = torch.tensor([[0.0]])  # 或者直接跳過這步

                action = int(torch.argmax(q_values))  # 選擇 Q 值最大的行動
                print(f"[Exploitation] ε={epsilon:.3f}, action={action}")
                
                print("q_values shape:", q_values.shape)
                print("actions:", actions)


            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = adjusted_reward(reward, done)

            print(state, action, reward, next_state, done)
            time.sleep(1.5)


            # 儲存經驗
            transition = (state, action, reward, next_state, done)
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            if buffer.length() < buffer_capacity + 1:
                continue

            # 更新量子電路
            transitions = buffer.sample(batch_size=10)
            states, actions, rewards, next_states, dones = zip(*transitions)

            # 確保 dones 是 tensor 並轉為 float32
            dones = torch.tensor(dones, dtype=torch.float32)

            # 從 q_values 中選擇相應 action 的 Q 值
            q_values_selected = torch.gather(q_values, 1, torch.tensor(actions).unsqueeze(1))

            # 計算 TD 目標（時間差分目標）
            td_targets = reward + gemma * (1 - dones) * torch.max(q_values_selected)

            # 計算損失
            loss = torch.nn.MSELoss()(q_values_selected.squeeze(), td_targets)

            # 反向傳播並更新量子電路的參數
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # 更新 epsilon
        #epsilon = epsilon / (100 / episode + 1)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)




if __name__ == "__main__":
    deep_Q_Learning()
    print("Training completed.")
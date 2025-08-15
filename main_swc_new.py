import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
from collections import deque

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

# 環境設定
env_name = "FrozenLake-v1"
env = gym.make(env_name, render_mode="human", is_slippery=False)
state_space_size = env.observation_space.n
n_qubits = math.ceil(math.log2(state_space_size))
n_layers = 2
n_params = n_qubits * n_layers * 2
gemma = 0.99
seed = 42

# 建立參數化量子電路
def create_quantum_circuit():
    input_params = ParameterVector("x", n_qubits)
    weight_params = ParameterVector("w", n_params)
    qc = QuantumCircuit(n_qubits)

    # encoding
    for i in range(n_qubits):
        qc.rx(input_params[i], i)
    qc.barrier()
    for num in range(n_layers):
        # entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        # variational
        for i in range(n_qubits):
            qc.ry(weight_params[i + 2 * num * n_qubits], i)
            qc.rz(weight_params[(i + n_qubits) + 2 *  num * n_qubits], i)
        qc.barrier()
    print(qc.draw())
    return qc, input_params, weight_params

# PyTorch 模型包裝（使用 V2 primitive）
class QuantumQValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        qc, input_params, weight_params = create_quantum_circuit()
        sampler = StatevectorSampler()
        qnn = SamplerQNN(
            sampler=sampler,
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            input_gradients=True,
        )
        self.q_layer = TorchConnector(qnn)

    def forward(self, x):
        return self.q_layer(x)

# 經驗回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def length(self):
        return len(self.buffer)

#調整 reward
def manhattan_distance(state, goal=15):
    x1, y1 = divmod(state, 4)
    x2, y2 = divmod(goal, 4)
    return abs(x1 - x2) + abs(y1 - y2)

def adjusted_reward(reward, done, state, step, max_steps):
    if reward == 0:
        if done:
            return -5.0
        else:
            dist = manhattan_distance(state)
            return -0.01 + 0.02 * (1 - dist / 6)  # 根據距離終點給微弱正向回饋
    else:
        return 10.0

# def adjusted_reward(reward, done, state, step, max_steps):
#     if reward == 1:
#         return 10.0  # 成功到達終點
#     elif done:
#         return -5.0  # 掉進洞
#     else:
#         dist = manhattan_distance(state)
#         return -0.01 + 0.02 * (1 - dist / 6)  # 根據距離終點給微弱正向回饋

# 主訓練函數
def deep_Q_Learning():
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    buffer = ReplayBuffer(capacity=80)
    model = QuantumQValueModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    env.reset(seed=seed)
    for episode in range(1, 101):  # 可調整訓練回合數
        state, _ = env.reset()
        done = False
        total_reward = 0

        for step in range(100):
            state_bits = [float(b) for b in format(state, f'0{n_qubits}b')]
            state_tensor = torch.tensor(state_bits, dtype=torch.float32).unsqueeze(0)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                #q_values = torch.zeros((1, env.action_space.n))
            else:
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values))
                
            # 防呆：確保 action 在合法範圍
            action = int(action) % env.action_space.n

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            #reward = adjusted_reward(reward, done)
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            reward = adjusted_reward(reward, done, state, step, 100)
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done or buffer.length() < 32:
                break

            transitions = buffer.sample(10)
            states, actions, rewards, next_states, dones = zip(*transitions)
            state_batch = torch.tensor([[float(b) for b in format(s, f'0{n_qubits}b')] for s in states], dtype=torch.float32)
            q_values_batch = model(state_batch)
            actions_tensor = torch.tensor(actions).unsqueeze(1)
            q_values_selected = torch.gather(q_values_batch, 1, actions_tensor)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            td_targets = rewards_tensor + gemma * (1 - dones_tensor) * torch.max(q_values_selected, dim=1).values
            loss = nn.MSELoss()(q_values_selected.squeeze(), td_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode:03d} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    deep_Q_Learning()
    print("Training completed.")
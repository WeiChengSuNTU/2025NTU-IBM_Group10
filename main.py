########################################################################################################################

# Updated version of https://github.com/ycchen1989/Var-QuantumCircuits-DeepRL/blob/master/Code/QML_DQN_FROZEN_LAKE.py

########################################################################################################################


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import torch
import torch.nn as nn
from torch.autograd import Variable

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import gymnasium as gym
import time
import random
from collections import namedtuple

from gymnasium.envs.registration import register

register(
    id='Deterministic-ShortestPath-4x4-FrozenLake-v0',  # name given to this new environment
    entry_point='ShortestPathFrozenLake:ShortestPathFrozenLake',  # env entry point
    kwargs={'map_name': '4x4', 'is_slippery': False}  # argument passed to the env
)

dtype = torch.DoubleTensor

########################################################################################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def output_all(self):
        return self.memory

    def __len__(self):
        return len(self.memory)


########################################################################################################################


## Plotting Function ##
"""
Note: the plotting code is origin from Yang, Chao-Han Huck, et al. "Enhanced Adversarial Strategically-Timed Attacks Against Deep Reinforcement Learning." 
## ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2020.
If you use the code in your research, please cite the original reference. 
"""


def plotTrainingResultCombined(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
    fig, ax = plt.subplots()
    # plt.yscale('log')
    ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
    ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
    leg = ax.legend();

    ax.set(xlabel='Iteration Index',
           title=_fileTitle)
    fig.savefig(_fileTitle + "_" + datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")


def plotTrainingResultReward(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
    fig, ax = plt.subplots()
    # plt.yscale('log')
    ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
    # ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
    leg = ax.legend();

    ax.set(xlabel='Iteration Index',
           title=_fileTitle)
    fig.savefig(_fileTitle + "_REWARD" + "_" + datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")


########################################################################################################################

def decimalToBinaryFixLength(_length, _decimal):
    binNum = bin(int(_decimal))[2:]
    outputNum = [int(item) for item in binNum]
    if len(outputNum) < _length:
        outputNum = np.concatenate((np.zeros((_length - len(outputNum),)), np.array(outputNum)))
    else:
        outputNum = np.array(outputNum)
    return outputNum

########################################################################################################################
## qiskit Part ##
num_qubits = 4
num_layers = 2
simulator = AerSimulator()

def state_preparation_qiskit(a):
    qc = QuantumCircuit(num_qubits)
    for ind in range(len(a)):
        val_float = float(a[ind])
        qc.rx(np.pi * val_float, ind)
        qc.rz(np.pi * val_float, ind)
    return qc


def layer_qiskit(qc, weights):
    # CNOT entanglement
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    for i in range(num_qubits):
        w0, w1, w2 = map(float, weights[i])
        qc.rz(w0, i)
        qc.ry(w1, i)
        qc.rz(w2, i)
    return qc


def circuit_qiskit(weights, angles=None):
    a = angles if angles is not None else np.zeros(num_qubits)
    qc = state_preparation_qiskit(a)

    for W in weights:
        qc = layer_qiskit(qc, W)

    qc.save_statevector()
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc).result()
    statevector = result.get_statevector()

    # Ensure statevector is a NumPy array
    statevector = np.array(statevector)

    # Expectation values <Z> for each qubit
    expvals = []
    for i in range(num_qubits):
        Z = np.array([[1, 0], [0, -1]])
        op = 1
        for j in range(num_qubits):
            op = np.kron(op, Z if j == i else np.eye(2))
        expval = np.real(np.vdot(statevector, op @ statevector))
        expvals.append(expval)

    return np.array(expvals)


def variational_classifier(var_Q_circuit, var_Q_bias, angles=None):
    """The variational classifier."""
    raw_output = torch.tensor(
        circuit_qiskit(var_Q_circuit.detach().numpy(), angles=angles)
    )
    raw_output = raw_output + var_Q_bias
    return raw_output

########################################################################################################################

def square_loss(labels, predictions):
    """ Square loss function

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: square loss
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    #print("LOSS")
    #print(loss)

    return loss


def cost(var_Q_circuit, var_Q_bias, features, labels):
    """Cost (error) function to be minimized."""

    # predictions = [variational_classifier(weights, angles=f) for f in features]
    # Torch data type??

    predictions = [variational_classifier(var_Q_circuit=var_Q_circuit, var_Q_bias=var_Q_bias,
                                          angles=decimalToBinaryFixLength(4, item.state))[item.action] for item in
                   features]
    predictions = torch.tensor(predictions, requires_grad=True)
    labels = torch.tensor(labels)
    #print("PRIDICTIONS:")
    #print(predictions)
    #print("LABELS:")
    #print(labels)

    return square_loss(labels, predictions)


########################################################################################################################

def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """

    # Modify to incorporate with Variational Quantum Classifier
    # epsilon should change along training
    # In the beginning => More Exploration
    # In the end => More Exploitation

    # More Random
    np.random.seed(int(datetime.now().strftime("%S%f")))

    if train or np.random.rand() < ((epsilon / n_actions) + (1 - epsilon)):
        # action = np.argmax(Q[s, :])
        # variational classifier output is torch tensor
        # action = np.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(9,s)))
        action = torch.argmax(variational_classifier(var_Q_circuit=var_Q_circuit, var_Q_bias=var_Q_bias,
                                                     angles=decimalToBinaryFixLength(4, s)))
    else:
        # need to be torch tensor
        action = torch.tensor(np.random.randint(0, n_actions))
    return action

########################################################################################################################

def deep_Q_Learning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render=False, test=False):
    """
    Updated for Gymnasium (reset()/step() return signature changed).
    """
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, render_mode="ansi", is_slippery=False)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    print("NUMBER OF STATES:" + str(n_states))
    print("NUMBER OF ACTIONS:" + str(n_actions))

    # Q circuit init
    num_qubits = 4
    num_layers = 2

    var_init_circuit = Variable(
        torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
    var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

    var_Q_circuit = var_init_circuit
    var_Q_bias = var_init_bias

    var_target_Q_circuit = var_Q_circuit.clone().detach()
    var_target_Q_bias = var_Q_bias.clone().detach()

    opt = torch.optim.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08)

    TARGET_UPDATE = 25
    batch_size = 5
    OPTIMIZE_STEPS = 5

    target_update_counter = 0

    iter_index = []
    iter_reward = []
    iter_total_steps = []

    cost_list = []
    timestep_reward = []

    memory = ReplayMemory(80)

    for episode in range(episodes):
        print(f"Episode: {episode}")
        # Gymnasium reset returns (obs, info)
        s, _ = env.reset()
        s = int(s)  # ensure python int for decimalToBinaryFixLength
        a = epsilon_greedy(var_Q_circuit=var_Q_circuit, var_Q_bias=var_Q_bias, epsilon=epsilon, n_actions=n_actions,
                           s=s).item()
        t = 0
        total_reward = 0
        done = False

        while t < max_steps:
            if render:
                # If render_mode="ansi" env.render() returns a string
                rendered = env.render()
                # print rendered output if any
                if rendered is not None:
                    print(rendered)

            t += 1
            target_update_counter += 1

            # Gymnasium step signature: obs, reward, terminated, truncated, info
            s_, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            s_ = int(s_)  # ensure it is an int

            if done and reward == 0:
                reward -= 0.2
            else:
                if s == s_:
                    reward -= 0.1
                else:
                    reward -= 0.01

            total_reward += reward
            a_ = epsilon_greedy(var_Q_circuit=var_Q_circuit, var_Q_bias=var_Q_bias, epsilon=epsilon,
                                n_actions=n_actions, s=s_).item()

            memory.push(s, a, reward, s_, done)

            if len(memory) > batch_size:
                batch_sampled = memory.sample(batch_size=batch_size)

                # build target Q values using target network
                Q_target = [
                    item.reward + (1 - int(item.done)) * gamma * torch.max(
                        variational_classifier(var_Q_circuit=var_target_Q_circuit,
                                               var_Q_bias=var_target_Q_bias,
                                               angles=decimalToBinaryFixLength(4, int(item.next_state))))
                    for item in batch_sampled
                ]

                # optimization step
                def closure():
                    opt.zero_grad()
                    loss = cost(var_Q_circuit=var_Q_circuit, var_Q_bias=var_Q_bias, features=batch_sampled,
                                labels=Q_target)
                    loss.backward()
                    return loss

                opt.step(closure)

                # optional: compute loss across entire replay (if you want)
                # current_replay_memory = memory.output_all()
                # current_target_for_replay_memory = [...]

            if target_update_counter > TARGET_UPDATE:
                print("UPDATING TARGET CIRCUIT...")
                var_target_Q_circuit = var_Q_circuit.clone().detach()
                var_target_Q_bias = var_Q_bias.clone().detach()
                target_update_counter = 0

            s, a = s_, a_

            if done:
                if render:
                    rendered = env.render()
                    if rendered is not None:
                        print(rendered)
                # decay epsilon
                epsilon = epsilon / ((episode / 100) + 1)
                print(f"This episode took {t} timesteps and reward: {total_reward}")
                timestep_reward.append(total_reward)
                iter_index.append(episode)
                iter_reward.append(total_reward)
                iter_total_steps.append(t)
                break

    return timestep_reward, iter_index, iter_reward, iter_total_steps, var_Q_circuit, var_Q_bias


def test_agent(Q, env, n_tests, delay=1):
    """Updated test agent for Gymnasium API."""
    n_states, n_actions = env.observation_space.n, env.action_space.n
    for test in range(n_tests):
        print(f"Test #{test}")
        s, _ = env.reset()
        s = int(s)
        done = False
        epsilon = 0  # fully greedy
        while True:
            time.sleep(delay)
            rendered = env.render()
            if rendered is not None:
                print(rendered)
            a = epsilon_greedy(var_Q_circuit=Q[0], var_Q_bias=Q[1], epsilon=epsilon, n_actions=n_actions, s=s,
                               train=True)  # adapt call if you store params differently
            a = int(a) if isinstance(a, (int, np.integer)) else a.item()
            print(f"Chose action {a} for state {s}")
            s_, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            s = int(s_)
            if done:
                if reward > 0:
                    print("Reached goal!")
                else:
                    print("Dead :(")
                time.sleep(1)
                break

########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    alpha = 0.4
    gamma = 0.9
    epsilon = 1.0
    episodes = 100
    max_steps = 2500
    n_tests = 2
    timestep_reward, iter_index, iter_reward, iter_total_steps, var_Q_circuit, var_Q_bias = deep_Q_Learning(alpha,
                                                                                                            gamma,
                                                                                                            epsilon,
                                                                                                            episodes,
                                                                                                            max_steps,
                                                                                                            n_tests,
                                                                                                            render=False,
                                                                                                            test=False)


    file_title = 'VQDQN_Frozen_Lake_NonSlip_Dynamic_Epsilon_RMSProp' + datetime.now().strftime("NO%Y%m%d%H%M%S")
    with open(file_title + "_var_Q_circuit" + ".txt", "wb") as fp:
        pickle.dump(var_Q_circuit, fp)
    with open(file_title + "_var_Q_bias" + ".txt", "wb") as fp:
        pickle.dump(var_Q_bias, fp)
    with open(file_title + "_iter_reward" + ".txt", "wb") as fp:
        pickle.dump(iter_reward, fp)
    print("Trained agent saved. Now testing...")


    env_name = "FrozenLake-v1"
    env = gym.make(env_name, render_mode="ansi", is_slippery=False)
    test_agent(
        Q=(var_Q_circuit, var_Q_bias),
        env=env,
        n_tests=3,
    )

    print(timestep_reward)

    ## Drawing Training Result ##
    plotTrainingResultReward(_iter_index=iter_index, _iter_reward=iter_reward, _iter_total_steps=iter_total_steps,
                             _fileTitle='Quantum_DQN_Frozen_Lake_NonSlip_Dynamic_Epsilon_RMSProp')

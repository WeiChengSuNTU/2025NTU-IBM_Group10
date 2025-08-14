import gymnasium as gym
import pygame
import time
import random


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


def deep_Q_Learning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
	"""
	@param alpha learning rate
	@param gamma decay factor
	@param epsilon for exploration
	@param max_steps for max step in each episode
	@param n_tests number of test episodes
	"""


    seed = 42

	env_name = "FrozenLake-v1"
    env = gym.make(env_name, render_mode="human", is_slippery=False)

	n_states, n_actions = env.observation_space.n, env.action_space.n
	print("NUMBER OF STATES:" + str(n_states))
	print("NUMBER OF ACTIONS:" + str(n_actions))

	# Initialize Q function approximator variational quantum circuit
	# initialize weight layers

	num_qubits = 4
	num_layers = 2
	# var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
	var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

	# Define the two Q value function initial parameters
	# Use np copy() function to DEEP COPY the numpy array
	var_Q_circuit = var_init_circuit
	var_Q_bias = var_init_bias
	# print("INIT PARAMS")
	# print(var_Q_circuit)

	var_target_Q_circuit = var_Q_circuit.clone().detach()
	var_target_Q_bias = var_Q_bias.clone().detach()

	##########################
	# Optimization method => random select train batch from replay memory
	# and opt

	# opt = NesterovMomentumOptimizer(0.01)

	# opt = torch.optim.Adam([var_Q_circuit, var_Q_bias], lr = 0.1)
	# opt = torch.optim.SGD([var_Q_circuit, var_Q_bias], lr=0.1, momentum=0.9)
	opt = torch.optim.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

	## NEed to move out of the function
	TARGET_UPDATE = 20
	batch_size = 5
	OPTIMIZE_STEPS = 5
	##


	target_update_counter = 0

	iter_index = []
	iter_reward = []
	iter_total_steps = []

	cost_list = []


	timestep_reward = []


	# Demo of generating a ACTION
	# Output a numpy array of value for each action

	# Define the replay memory
	# Each transition:
	# (s_t_0, a_t_0, r_t, s_t_1, 'DONE')

	memory = ReplayMemory(80)

	# Input Angle = decimalToBinaryFixLength(9, stateInd)
	# Input Angle is a numpy array

	# stateVector = decimalToBinaryFixLength(9, stateInd)

	# q_val_s_t = variational_classifier(var_Q, angles=stateVector)
	# # action_t = q_val_s_t.argmax()
	# action_t = epsilon_greedy(var_Q, epsilon, n_actions, s)
	# q_val_target_s_t = variational_classifier(var_target_Q, angles=stateVector)

	# train the variational classifier


	for episode in range(episodes):
		print(f"Episode: {episode}")
		# Output a s in decimal format
		s = env.reset()
		# Doing epsilog greedy action selection
		# With var_Q
		a = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s).item()
		t = 0
		total_reward = 0
		done = False


		while t < max_steps:
			if render:
				print("###RENDER###")
				env.render()
				print("###RENDER###")
			t += 1

			target_update_counter += 1

			# Execute the action 
			s_, reward, done, info = env.step(a)
			# print("Reward : " + str(reward))
			# print("Done : " + str(done))
			total_reward += reward
			# a_ = np.argmax(Q[s_, :])
			a_ = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s_).item()
			
			# print("ACTION:")
			# print(a_)

			memory.push(s, a, reward, s_, done)

			if len(memory) > batch_size:

				# Sampling Mini_Batch from Replay Memory

				batch_sampled = memory.sample(batch_size = batch_size)

				# Transition = (s_t, a_t, r_t, s_t+1, done(True / False))

				# item.state => state
				# item.action => action taken at state s
				# item.reward => reward given based on (s,a)
				# item.next_state => state arrived based on (s,a)

				Q_target = [item.reward + (1 - int(item.done)) * gamma * torch.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=decimalToBinaryFixLength(4,item.next_state))) for item in batch_sampled]
				# Q_prediction = [variational_classifier(var_Q, angles=decimalToBinaryFixLength(9,item.state))[item.action] for item in batch_sampled ]

				# Gradient Descent
				# cost(weights, features, labels)
				# square_loss_training = square_loss(labels = Q_target, Q_predictions)
				# print("UPDATING PARAMS...")

				# CHANGE TO TORCH OPTIMIZER
				
				# var_Q = opt.step(lambda v: cost(v, batch_sampled, Q_target), var_Q)
				# opt.zero_grad()
				# loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
				# print(loss)
				# FIX this gradient error
				# loss.backward()
				# opt.step(loss)

				def closure():
					opt.zero_grad()
					loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
					# print(loss)
					loss.backward()
					return loss
				opt.step(closure)

				# print("UPDATING PARAMS COMPLETED")
				current_replay_memory = memory.output_all()
				current_target_for_replay_memory = [item.reward + (1 - int(item.done)) * gamma * torch.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=decimalToBinaryFixLength(4,item.next_state))) for item in current_replay_memory]
				# current_target_for_replay_memory = [item.reward + (1 - int(item.done)) * gamma * np.max(variational_classifier(var_target_Q, angles=decimalToBinaryFixLength(9,item.next_state))) for item in current_replay_memory]

				# if t%5 == 0:
				# 	cost_ = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = current_replay_memory, labels = current_target_for_replay_memory)
				# 	print("Cost: ")
				# 	print(cost_.item())
				# 	cost_list.append(cost_)


			if target_update_counter > TARGET_UPDATE:
				print("UPDATEING TARGET CIRCUIT...")

				var_target_Q_circuit = var_Q_circuit.clone().detach()
				var_target_Q_bias = var_Q_bias.clone().detach()
				
				target_update_counter = 0

			s, a = s_, a_

			if done:
				if render:
					print("###FINAL RENDER###")
					env.render()
					print("###FINAL RENDER###")
					print(f"This episode took {t} timesteps and reward: {total_reward}")
				epsilon = epsilon / ((episode/100) + 1)
				# print("Q Circuit Params:")
				# print(var_Q_circuit)
				print(f"This episode took {t} timesteps and reward: {total_reward}")
				timestep_reward.append(total_reward)
				iter_index.append(episode)
				iter_reward.append(total_reward)
				iter_total_steps.append(t)
				break
	# if render:
	# 	print(f"Here are the Q values:\n{Q}\nTesting now:")
	# if test:
	# 	test_agent(Q, env, n_tests, n_actions)
	return timestep_reward, iter_index, iter_reward, iter_total_steps, var_Q_circuit, var_Q_bias

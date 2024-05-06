import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import gymnasium, collections
import pandas as pd

def createDNN( nInputs, nOutputs, nLayer, nNodes ):
	"""
	Function that generates a neural network with the given requirements.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	# Initialize the neural network
	model = Sequential()
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	for _ in range(nLayer):
		model.add(Dense(nNodes, activation="relu")) 
	model.add(Dense(nOutputs, activation="linear")) 
	model.add(Dense(nOutputs, activation="softmax")) 
	#
	return model

def createValueDNN( nInputs=4, nOutputs=1, nLayer=1, nNodes=64 ):
	"""
	Function that generates a neural network with the given requirements.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	# Initialize the value function neural network
	model = Sequential()
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	for _ in range(nLayer):	
		model.add(Dense(nNodes, activation="relu")) 
	model.add(Dense(nOutputs, activation="linear")) 

	return model

class TorchModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""
	
	# Initialize the neural network
	def __init__(self, nInputs, nOutputs, nLayer, nNodes):
		
		super(TorchModel, self).__init__()
		self.nLayer = nLayer

		# input layer
		self.fc1 = nn.Linear(nInputs, nNodes)


		#hidden layers
		for i in range(nLayer):
			layer_name = f"fc{i+2}"
			self.add_module(layer_name, nn.Linear(nNodes, nNodes))  

		#output
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		#return F.softmax(x, dim=1)
		return F.softmax(x)
	


def mse(predicted_value, target):
	"""
	Compute the MSE loss function

	"""
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse
	
class ValueModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""
	
	# Initialize the neural network
	def __init__(self, nInputs=4, nOutputs=1, nLayer=1, nNodes=64):
		
		super(ValueModel, self).__init__()
		self.nLayer = nLayer

		# input layer
		self.fc1 = nn.Linear(nInputs, nNodes)

		#hidden layers
		for i in range(nLayer):
			layer_name = f"fc{i+2}"
			self.add_module(layer_name, nn.Linear(nNodes, nNodes))  

		#output
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		return x



def training_loop(env, neural_net, updateRule, keras=True, total_episodes=1500, gamma=0.99, baseline=False):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the trainign.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	# Reset the global optimizer and memories before the training
	optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5) if keras else optim.Adam(neural_net.parameters(), lr=4e-5)
	if baseline: 
		value_net = createValueDNN() if keras else ValueModel()
		optimizer_v = tf.keras.optimizers.Adam(learning_rate=4e-5) if keras else optim.Adam(value_net.parameters(), lr=4e-5)
	

	rewards_list, reward_queue = [], collections.deque(maxlen=100)
	memory_buffer = []
	for episode in range(total_episodes):

		# Reset the environment and the episode reward before the episode
		#state = None
		state = env.reset()[0]
		ep_reward = 0
		#memory_buffer.append([])

		while True:

			# Select the action to perform
			if not keras:
				distribution = neural_net(torch.tensor(state).reshape(-1, 4)).detach().numpy()[0]

			action = np.random.choice(env.action_space.n, p=distribution)
		
			# Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated

			memory_buffer.append([state, action, reward, next_state, done])

			ep_reward += reward

			# Exit condition for the episode
			if done: break
			state = next_state

		# Update the reward list to return
		reward_queue.append(ep_reward)
		rewards_list.append(np.mean(reward_queue))
		print( f"episodes {episode:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f})" )

		
		# An episode is over,then update
		if not baseline:
			REINFORCE(neural_net, keras, memory_buffer, gamma, optimizer, baseline)
		else:
			value_net = createValueDNN() if keras else ValueModel()
			REINFORCE(neural_net, keras, memory_buffer, gamma, optimizer, baseline, value_net, optimizer_v)
		
		# clean the memory buffer
		memory_buffer = []

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list


def REINFORCE(neural_net, keras, memory_buffer, gamma, optimizer, baseline, value_net=None, optimizer_v=None):

	"""
	Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

	"""

	states = []
	actions = []
	rewards = []
	
	for ep in range(len(memory_buffer)):
		# Extraction of the information from the buffer (for the considered episode)
		# memory_buffer.append([state, action, reward, next_state, done])
		states.append(memory_buffer[ep][0])
		actions.append(memory_buffer[ep][1])
		rewards.append(memory_buffer[ep][2])

	# Iterate over all the trajectories considered
 	# calculate the return G reversely using reward-to-go tecnique
	G =  []
	g = 0

	for r in reversed(rewards):
		g = gamma * g + r
		G.insert(0, g)	
	

	if not baseline:
		if not keras:
			for t in range(len(rewards)):
				state = states[t]
				action = actions[t]
				g = G[t]

				a_prob = neural_net(torch.tensor(state))
				policy_loss = -gamma**t * g * np.log(a_prob[action])
				optimizer.zero_grad()
				policy_loss.backward()
				optimizer.step()
		else:
			for t in range(len(rewards)):
				state = None
				action = None
				g = None

				with tf.GradientTape() as tape:
					policy_loss = None #TODO

				grad = None #TODO
				... # TODO

	else:
		if not keras: 
			for t in range(len(rewards)):
				state = states[t]
				action = actions[t]
				g = G[t]

				v_s = value_net(torch.tensor(state))
				
				a_prob = neural_net(torch.tensor(state))
				policy_loss = -gamma**t * (g-v_s) * torch.log(a_prob[action])
				
				optimizer.zero_grad()
				policy_loss.backward(retain_graph=True)
				optimizer.step()

				# Update value function
				value_loss = F.mse_loss(v_s, torch.tensor([g]))
				optimizer_v.zero_grad()
				value_loss.backward()
				optimizer_v.step()

		else:
			for t in range(len(rewards)):
				state = None # TODO
				action = None #TODO
				g = None #TODO
				
				with tf.GradientTape() as value_tape:
					v_s = None #TODO
					value_loss = mse(v_s, g)

				grad_vf = None #TODO
				#TODO
				...
				
				with tf.GradientTape() as policy_tape:
					a_prob = None #TODO
					policy_loss = None # TODO
				
				grad = None #TODO
				#TODO
				...


			
def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
	print( "*                 (REINFORCE)                   *" )
	print( "*************************************************\n" )

	training_episodes = 1200
	number_seeds_to_test = 3
	gamma=0.99
	
	# setting DNN configuration
	nInputs=4
	nOutputs=2
	nLayer=2
	nNodes=32
	use_torch = True 
 
	if use_torch:
		print("\nTraining torch model using REINFORCE baseline...\n")
		rewards_torch_baseline = []
		for _ in range(number_seeds_to_test):
			env = gymnasium.make("CartPole-v1")#, render_mode="human" )
			neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes)
			rewards_torch_baseline.append(training_loop(env, neural_net_torch, REINFORCE, keras=False, total_episodes=training_episodes, gamma=gamma, baseline=True))

		print("\nTraining torch model using REINFORCE...\n")
		rewards_torch_naive = []
		for _ in range(number_seeds_to_test):
			env = gymnasium.make("CartPole-v1")#, render_mode="human" )
			neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes)
			rewards_torch_naive.append(training_loop(env, neural_net_torch, REINFORCE, keras=False, total_episodes=training_episodes, gamma=gamma, baseline=False))

		
		# plotting the results
		t = list(range(0, training_episodes))

		data_torch = {'Environment Step': [], 'Mean Reward': []}
		for _, rewards in enumerate(rewards_torch_naive):
			for step, reward in zip(t, rewards):
				data_torch['Environment Step'].append(step)
				data_torch['Mean Reward'].append(reward)
		df_torch = pd.DataFrame(data_torch)

		data_torch_baseline = {'Environment Step': [], 'Mean Reward': []}
		for _, rewards in enumerate(rewards_torch_baseline):
			for step, reward in zip(t, rewards):
				data_torch_baseline['Environment Step'].append(step)
				data_torch_baseline['Mean Reward'].append(reward)
		df_torch_baseline = pd.DataFrame(data_torch_baseline)

		
		# Plotting
		sns.set_style("darkgrid")
		#sns.color_palette("Set2")
		plt.figure(figsize=(8, 6))  # Set the figure size
		sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='REINFORCE', errorbar='se')
		sns.lineplot(data=df_torch_baseline, x='Environment Step', y='Mean Reward', label='REINFORCE_baseline', errorbar='se')

		# Add title and labels
		plt.title('Comparison REINFORCE vs REINFORCE_baseline PyTorch on CartPole-v1')
		plt.xlabel('Episodes')
		plt.ylabel('Mean Reward')

		# Show legend
		plt.legend()

		# Show plot
		plt.show()
			

	else:

		print("\nTraining keras model using REINFORCE baseline...\n")
		rewards_keras_baseline = []
		for _ in range(number_seeds_to_test):
			env = gymnasium.make("CartPole-v1")#, render_mode="human" )
			neural_net_keras = createDNN(nInputs, nOutputs, nLayer, nNodes)
			rewards_keras_baseline.append(training_loop(env, neural_net_keras, REINFORCE, keras=True, total_episodes=training_episodes, gamma=gamma, baseline=True))

		print("\nTraining keras model using REINFORCE...\n")
		rewards_keras_naive = []
		for _ in range(number_seeds_to_test):
			env = gymnasium.make("CartPole-v1")#, render_mode="human" )
			neural_net_keras = createDNN(nInputs, nOutputs, nLayer, nNodes)
			rewards_keras_naive.append(training_loop(env, neural_net_keras, REINFORCE, keras=True, total_episodes=training_episodes, gamma=gamma, baseline=False))


		data_keras = {'Environment Step': [], 'Mean Reward': []}
		for _, rewards in enumerate(rewards_keras_naive):
			for step, reward in zip(t, rewards):
				data_keras['Environment Step'].append(step)
				data_keras['Mean Reward'].append(reward)
		df_keras = pd.DataFrame(data_keras)

		data_keras_baseline = {'Environment Step': [], 'Mean Reward': []}
		for _, rewards in enumerate(rewards_keras_baseline):
			for step, reward in zip(t, rewards):
				data_keras_baseline['Environment Step'].append(step)
				data_keras_baseline['Mean Reward'].append(reward)
		df_keras_baseline = pd.DataFrame(data_keras_baseline)

			
		# Plotting
		sns.set_style("darkgrid")
		#sns.color_palette("Set2")
		plt.figure(figsize=(8, 6))  # Set the figure size
		sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='REINFORCE', errorbar='se')
		sns.lineplot(data=df_keras_baseline, x='Environment Step', y='Mean Reward', label='REINFORCE_baseline', errorbar='se')

		# Add title and labels
		plt.title('Comparison REINFORCE vs REINFORCE_baseline Keras on CartPole-v1')
		plt.xlabel('Episodes')
		plt.ylabel('Mean Reward')

		# Show legend
		plt.legend()

		# Show plot
		plt.show()
			



if __name__ == "__main__":
	main()	

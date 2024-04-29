import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import pandas as pd


def createDNN_keras(nInputs, nOutputs, nLayer, nNodes):
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
	#
	# YOUR CODE HERE!
	#
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
		self.fc1 = nn.Linear(nInputs, nNodes)
		
		self.hidden_layers = nLayer
		for i in range(nLayer):
			setattr(self, f"fc{i+2}", nn.Linear(nNodes, nNodes))

		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))

		for i in range(self.hidden_layers):
			x = F.relu(getattr(self, f"fc{i+2}")(x))

		x = self.output(x)

		return x

def mse(network, dataset_input, target):
	"""
	Compute the MSE loss function

	"""
	
	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network(dataset_input)
	
	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)
	
	# Return the averaged values for computational optimization
	return mse


def training_loop(env, neural_net, updateRule, keras=True, eps=1.0, updates=1, episodes=100):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	#TODO: initialize the optimizer 
	if keras:
		optimizer = None
	else:
		optimizer = optim.SGD(neural_net.parameters(), lr=0.01)

	 
	rewards_list, memory_buffer = [], collections.deque( maxlen=1000 )
	averaged_rewards = []
	for ep in range(episodes):

		state = env.reset()[0]

		ep_reward = 0

		while True:

			#TODO: select the action to perform exploiting an epsilon-greedy strategy
			#action = env.action_space.sample() 
			if np.random.rand() < eps:
				if keras:
					pass
				else:
					action = neural_net(torch.tensor(state)).argmax().item()
			else:
				action = np.random.randint(2)

			#TODO: update epsilon value
			#eps *= ...
			eps *= 0.99
   
			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated

			memory_buffer.append([state, action, reward, next_state, done])
			ep_reward += reward

			# Perform the actual training
			for _ in range(updates):
				updateRule(neural_net, keras, memory_buffer, optimizer)
				

			#TODO: modify the exit condition for the episode
			if done: break

			#TODO: update the current state
			state = next_state

		# Update the reward list to return
		rewards_list.append(ep_reward)
		averaged_rewards.append(np.mean(rewards_list))
		print( f"episode {ep:2d}: mean reward: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}" )
		

	# Close the enviornment and return the rewards list
	env.close()
	return averaged_rewards


def DQNupdate(neural_net, keras, memory_buffer, optimizer, batch_size=32, gamma=0.99):

	"""
	Main update rule for the DQN process. Extract data from the memory buffer and update 
	the newtwork computing the gradient.

	"""

	if len(memory_buffer) < batch_size: return

	indices = np.random.randint( len(memory_buffer), size=batch_size)
	for idx in indices: 

		state, action, reward, next_state, done = memory_buffer[idx]

		#TODO: compute the target for the training
		if keras:
			target = None
		else:
			optimizer.zero_grad()
			predicted = neural_net(torch.tensor(state))
			target = torch.clone(predicted)


		#TODO: update target using the update rule...
		if done:
			target[action] = reward
		else:
			if keras:
				max_q = np.max(neural_net([next_state]))
			else:
				max_q = np.max(neural_net(torch.tensor(next_state)).detach().numpy())
			target[action] = reward + gamma * max_q


		#TODO: compute the gradient and perform the backpropagation step using the selected framework
		if keras:
			with tf.GradientTape() as tape:
				objective = mse(neural_net, state, target)

		else:
			#objective = F.mse_loss(target, torch.tensor(action).float())
			objective = F.mse_loss(predicted, target)
			objective.backward()
			optimizer.step()


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the eighth lesson of the RL-Lab!   *" )
	print( "*               (Deep Q-Network)                 *" )
	print( "**************************************************\n" )

	training_steps = 50
	
	# setting DNN configuration
	nInputs=4
	nOutputs=2
	nLayer=2
	nNodes=32 

	print("\nTraining torch model...\n")
	rewards_torch = []
	for _ in range(5):
		env = gymnasium.make("CartPole-v1")#, render_mode="human" )
		neural_net_torch = TorchModel(nInputs, nOutputs, nLayer, nNodes)
		rewards_torch.append(training_loop(env, neural_net_torch, DQNupdate, keras=False, episodes=training_steps))

	"""print("\nTraining keras model...\n")
	rewards_keras = []
	for _ in range(10):
		env = gymnasium.make("CartPole-v1")#, render_mode="human" )
		neural_net_keras = createDNN_keras(nInputs, nOutputs, nLayer, nNodes)
		rewards_keras.append(training_loop(env, neural_net_keras, DQNupdate, keras=True, episodes=training_steps))"""


	# plotting the results
	t = list(range(0, training_steps))

	data = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_torch):
		for step, reward in zip(t, rewards):
			data['Environment Step'].append(step)
			data['Mean Reward'].append(reward)
	df_torch = pd.DataFrame(data)

	"""
	data_keras = {'Environment Step': [], 'Mean Reward': []}
	for _, rewards in enumerate(rewards_keras):
		for step, reward in zip(t, rewards):
			data_keras['Environment Step'].append(step)
			data_keras['Mean Reward'].append(reward)
	df_keras = pd.DataFrame(data_keras)
	"""

	# Plotting
	sns.set_style("darkgrid")
	plt.figure(figsize=(8, 6))  # Set the figure size
	sns.lineplot(data=df_torch, x='Environment Step', y='Mean Reward', label='torch', errorbar='se')
	#sns.lineplot(data=df_keras, x='Environment Step', y='Mean Reward', label='keras', errorbar='se')

	# Add title and labels
	plt.title('Comparison Keras and PyTorch on CartPole-v1')
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.show()
	plt.savefig('comparison.pdf')


if __name__ == "__main__":
	main()	
import os, sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld
import numpy as np


def value_iteration(environment, maxiters=300, discount=0.9, max_error=1e-3):
	"""
	Performs the value iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		max_error: the maximum error allowd in the utility of any state
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	
	U_1 = [0 for _ in range(environment.observation_space)] # vector of utilities for states S
	delta = 0
	U = U_1.copy()

	while True:
		delta = 0 # maximum change in the utility o any state in an iteration
		U = U_1.copy()

		for state in range(environment.observation_space):
			U_1[state] = environment.R[state] + discount * max([
				sum([
					environment.transition_prob(state, action, state_1) * U[state_1]
					for state_1 in range(environment.observation_space)
				])
				for action in range(environment.action_space)
			])
			if abs(U_1[state] - U[state]) > delta:
				delta = abs(U_1[state] - U[state])

		if delta < max_error * (1 - discount) / discount:
			break

	return environment.values_to_policy( U )

	

def policy_iteration(environment, maxiters=300, discount=0.9, maxviter=10):
	"""
	Performs the policy iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		maxviter: number of epsiodes for the policy evaluation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	
	p = [0 for _ in range(environment.observation_space)] #initial policy    
	U = [0 for _ in range(environment.observation_space)] #utility array
	

	while True:

		for state in range(environment.observation_space):
			U[state] = environment.R[state] + discount * sum([
				environment.transition_prob(state, p[state], state_1) * U[state_1]
				for state_1 in range(environment.observation_space)
			])

		unchanged = True 

		for state in range(environment.observation_space):
			actions = [
				sum([
					environment.transition_prob(state, action, state_1) * U[state_1]
					for state_1 in range(environment.observation_space)
				])
				for action in range(environment.action_space)
			]
			if max(actions) > sum([
				environment.transition_prob(state, p[state], state_1) * U[state_1]
				for state_1 in range(environment.observation_space)
			]):
				p[state] = np.argmax(actions)
				unchanged = False
				
		if unchanged:
			break
	
	return p



def main():
	print( "\n************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!  *" )
	print( "*    (Policy Iteration and Value Iteration)    *" )
	print( "************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n1) Value Iteration:" )
	vi_policy = value_iteration( env )
	env.render_policy( vi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(vi_policy) )

	print( "\n2) Policy Iteration:" )
	pi_policy = policy_iteration( env )
	env.render_policy( pi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(pi_policy) )


if __name__ == "__main__":
	main()
import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ( environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99 ):
	"""
	Implements the DynaQ algorithm
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		n: steps for the planning phase
		eps: random value for the eps-greedy policy (probability of random action)
		alfa: step size for the Q-Table update
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""	

	Q = numpy.zeros((environment.observation_space, environment.action_space))
	M = numpy.array([[[None, None] for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
	
	for _ in range(maxiters):
		""" # CODED IN AN ALTERNATIVE WAY
		s = environment.random_initial_state()
		a = epsilon_greedy(Q, s, eps)
		s1 = environment.sample(a, s)
		r = environment.R[s1]

		Q[s, a] += alfa * ( r + gamma * Q[s1].max() - Q[s, a] )

		M[s, a] = [r, s1]

		for _ in range(n):
			idxs = numpy.where(numpy.all(M != None, axis=-1))
			idx = numpy.random.choice( len(idxs[0]) )
			s = idxs[0][idx]
			a = idxs[1][idx]
			[r, s1] = M[s, a]
			Q[s, a] += alfa * ( r + gamma * Q[s1].max() - Q[s, a] )
		"""
		s = environment.start_state
		while not environment.is_terminal(s):
			a = epsilon_greedy(Q, s, eps)
			s1 = environment.sample(a, s)
			r = environment.R[s1]

			Q[s, a] += alfa * ( r + gamma * Q[s1].max() - Q[s, a] )

			M[s, a] = [r, s1]
			
			s = s1

			for _ in range(n):
				idxs = numpy.where(numpy.all(M != None, axis=-1))
				idx = numpy.random.choice( len(idxs[0]) )
				temp_s = idxs[0][idx]
				a = idxs[1][idx]
				[r, s1] = M[temp_s, a]
				Q[temp_s, a] += alfa * ( r + gamma * Q[s1].max() - Q[temp_s, a] )
				



	policy = Q.argmax(axis=1) 
	return policy



def main():
	print( "\n************************************************" )
	print( "*   Welcome to the sixth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n6) Dyna-Q" )
	dq_policy_n00 = dynaQ( env, n=0  )
	dq_policy_n25 = dynaQ( env, n=25 )
	dq_policy_n50 = dynaQ( env, n=50 )

	env.render_policy( dq_policy_n50 )
	print()
	print( f"\tExpected reward with n=0 :", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected reward with n=50:", env.evaluate_policy(dq_policy_n50) )
	
	

if __name__ == "__main__":
	main()
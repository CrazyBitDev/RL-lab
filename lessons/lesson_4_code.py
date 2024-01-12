import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def on_policy_mc( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""

	p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	returns = [[[] for _ in range(environment.action_space)] for _ in range(environment.observation_space)]

	for s in range(environment.observation_space):
		action = numpy.random.randint(0, environment.action_space)
		for a in range(environment.action_space):
			if a == action:
				p[s][a] = 1 - eps + eps/environment.action_space
			else:
				p[s][a] = eps/environment.action_space

	for _ in range(maxiters):
		episodes = environment.sample_episode(p)
		G = 0
		for t in range(len(episodes)-2, -1, -1):

			st = episodes[t][0]
			at = episodes[t][1]

			G = gamma * G + episodes[t+1][2]

			"""found = False
			for previous_episode in episodes[:t]:
				if previous_episode[0] == st and previous_episode[1] == at:
					found = True
			
			if not found:"""

			returns[st][at].append(G)
			Q[st][at] = numpy.mean(returns[st][at])

			A = numpy.argmax(Q[st])

			for a in range(environment.action_space):
				p[st][a] = (1 - eps + eps / environment.action_space) if a == A else eps/environment.action_space
				


	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
	return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the fourth lesson of the RL-Lab!   *" )
	print( "*            (Monte Carlo RL Methods)            *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy" )
	mc_policy = on_policy_mc( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()

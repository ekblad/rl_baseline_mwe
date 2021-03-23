
# libraries:
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# environment and utils:
from folsom.folsom import FolsomEnv
from machinery import *
from agents import Planner

# based on example at: https://keras.io/examples/rl/ddpg_pendulum/

def main():

	DATA_DIR = Path.cwd()
	STOR_DIR = DATA_DIR / 'planner'
	STOR_DIR.mkdir(exist_ok=True)
	# OLD_DIR = DATA_DIR / 'planner_june1_3'
	OLD_DIR = None

	num_res_states = 2
	inflow_stack = 15 # 23 future inflow features at various time scales between 1 day and 5 years
	epi_steps = 40 * 365
	max_epi = 300
	eps = 0.1

	env = FolsomEnv(DATA_DIR,res_dim=(num_res_states+inflow_stack,),inflow_stack=inflow_stack,epi_length=epi_steps,ens='r1i1p1',) # policy: storage, inflows, day of year
	agent, env = Planner.create(env,weights_dir=OLD_DIR,max_epi=max_epi)#,eps=eps) # this has actor/critic and target networks contained within
	obs = env.reset(model='canesm2',ens='r1i1p1',)

	# Get the environment and extract the number of actions.

	env.seed(123) # not really used yet
	assert len(env.action_space.shape) == 1 # one continuous reservoir release as the action
	num_states = env.observation_space.shape
	print("Size of State Space ->  {}".format(num_states))
	num_actions = env.action_space.shape[0]
	print("Size of Action Space ->  {}".format(num_actions))
	num_res_states = env.reservoir_space.shape
	print("Size of Reservoir State Space ->  {}".format(num_res_states))

	upper_bound = env.action_space.high[0]
	lower_bound = env.action_space.low[0]

	print("Max Value of Action ->  {}".format(upper_bound))
	print("Min Value of Action ->  {}".format(lower_bound))

	# training hyperparameters
	std_dev = 1. 
	ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)) 

	# learning rate for actor-critic models:
	actor_lr = 5e-7 # slowest learner
	critic_lr = 10e-7 # should learn faster than actor

	# learning rate used to update target networks:
	tau = 10e-3 # the target actor and critic networks slowly take the actor/critic weights

	# initialize optimizers:
	critic_opt = tf.keras.optimizers.Adam(critic_lr)
	actor_opt = tf.keras.optimizers.Adam(actor_lr)

	# discount factor for future rewards
	gamma = 0.99

	batch_size = 1000
	buffer_capacity = 40*epi_steps
	buffer = Buffer(num_states,num_res_states,actor_opt,critic_opt,num_actions,buffer_capacity,batch_size,gamma)

	"""
	Implement main training loop, and iterate over episodes.
	Sample actions using `policy()` and train with `learn()` at each time step.
	Update the Target networks at a rate `tau`.
	"""
	env.epi_reward_list = []
	env.avg_reward_list = []
	env.avg_action_list = []

	# ensembles = ['r{}i1p1'.format(i) for i in np.arange(1,2)]
	ensembles = ['r1i1p1' for i in np.arange(0,10)]
	results = {} # not used yet
	epi_start = 30 * 365

	for ens in ensembles:
		print('Running data from ensemble member: {}'.format(ens))
		ens_done = False
		average_reward = 0
		average_action = 0

		while ens_done == False:
			prev_state = env.reset(model='canesm2',ens=ens,epi_start=epi_start)
			prev_res_state = prev_state
			episodic_reward = 0
			episodic_action = 0
			epi_done = False
			env.epi_count += 1

			while epi_done == False:
				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
				action, noise = agent.policy(tf_prev_state, ou_noise, agent.actor, env)
				state, reward, info = env.step(action, noise)
				res_state = state
				ens_done, epi_done = info['ens_done'], info['epi_done']
				episodic_reward += reward
				average_reward = average_reward+(reward-average_reward)/env.t
				average_action = average_action+(action[0]-average_action)/env.t				
				if env.t % 100 == 0:
					env.render(mode=['console'])
				buffer.record((prev_state, prev_res_state, action, reward, state, res_state))
				if env.epi_count > 1: # don't learn until at least 1 full batch in buffer
					buffer.learn(agent)
					update_target(agent.target_actor.variables, agent.actor.variables, tau)
					update_target(agent.target_critic.variables, agent.critic.variables, tau)
				prev_state = state
				prev_res_state = res_state

			env.epi_reward_list.append(episodic_reward)
			env.avg_reward_list.append(average_reward)
			env.avg_action_list.append(average_action)

			if env.epi_count % 10 == 0:
				# save the weights every n episodes
				agent.save_weights(env.epi_count,STOR_DIR)
				env.render(STOR_DIR=STOR_DIR,mode=['console','figures'])

			print("Episode * {} * Avg Reward is ==> {}".format(env.epi_count, np.mean(env.epi_reward_list[-40:])))
			if env.epi_count >= max_epi:
				exit()

if __name__ == '__main__':
	main()
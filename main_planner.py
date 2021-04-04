
# libraries:
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# environment and utils:
from folsom.folsom import Folsom
from machinery import *
from agents import Planner

# based on example at: https://keras.io/examples/rl/ddpg_pendulum/
# implements the three TD3 adjustments to original DDPG algorithm outlined in Fujimoto et al. (2018)

def main():

	DATA_DIR = Path.cwd()
	STOR_DIR = DATA_DIR / 'results_planner_6'
	STOR_DIR.mkdir(exist_ok=True)
	# OLD_DIR = DATA_DIR / 'results_planner_5'
	OLD_DIR = None

	num_res_states = 2 # day-of-year and reservoir storage
	inflow_stack = 15 # 15 future inflow features at various time scales between 1 day and 5 years
	TD3 = True # use TD3 adjustments to DDPG from Fujimoto et al. (2018)	
	warmup = True # use warmup period w/ random uniform action selection for one episode
	eps = 0.1 # if not None, use epsilon-greedy action exploration strategy
	epi_start = 30 * 365 # day of ensemble member to start episode on
	epi_steps = 40 * 365 # length of episode in days
	max_epi = 100 # upper bound used to determine annealing schedule (if not epsilon-greedy)	
	models = ['canesm2',]
	ensembles = ['r1i1p1' for i in np.arange(0,10)]

	env = Folsom(DATA_DIR,res_dim=(num_res_states+inflow_stack,),inflow_stack=inflow_stack,models=models,ensembles=ensembles,) # policy: storage, inflows, day of year
	agent, env = Planner.create(env,weights_dir=OLD_DIR,TD3=TD3,warmup=warmup,eps=eps,epi_start=epi_start,epi_steps=epi_steps,max_epi=max_epi) # this has actor/critic and target networks contained within
	obs = env.reset(agent)

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
	agent.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)) 

	# learning rate for actor-critic models:
	actor_lr = 10e-7 # slowest learner
	critic_lr = 10e-7 # should learn faster than actor

	# learning rate used to update target networks:
	tau = 10e-3 # the target actor and critic networks slowly take the actor/critic weights

	# initialize optimizers:
	clipnorm = 0.1
	# critic_opt = tf.keras.optimizers.SGD(critic_lr)#,clipnorm=clipnorm,momentum=0.9) # also sorta worked
	# actor_opt = tf.keras.optimizers.SGD(actor_lr)#,clipnorm=clipnorm,momentum=0.9)
	agent.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
	if TD3:
		agent.critic2_optimizer = tf.keras.optimizers.Adam(critic_lr)
	agent.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

	# discount factor for future rewards
	gamma = 0.99
	batch_size = 1000
	buffer_capacity = 100*epi_steps
	agent.buffer = Buffer(env, agent, buffer_capacity, batch_size, gamma)

	"""
	Implement main training loop, and iterate over episodes.
	Sample actions using `policy()` and train with `learn()` at each time step.
	Update the Target networks at a rate `tau`.
	"""

	# ensembles = ['r{}i1p1'.format(i) for i in np.arange(1,2)]
	ensembles = ['r1i1p1' for i in np.arange(0,10)]
	results = {} # not used yet

	if warmup:
		agent.buffer.warmup(env,agent,OLD_DIR)

	for ens in ensembles:
		print('Running data from ensemble member: {}'.format(ens))
		average_reward, average_action, ens_done = 0, 0, False
		while ens_done == False:
			prev_state = env.reset(agent)
			prev_res_state, episodic_reward, episodic_action, agent.epi_done = prev_state, 0, 0, False
			agent.epi_count += 1
			while agent.epi_done == False:
				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
				action, noise = agent.policy(tf_prev_state)
				state, reward, info = env.step(action, noise)
				res_state, agent.ens_done, agent.epi_done = state, info['ens_done'], info['epi_done']
				episodic_reward += reward
				average_reward = average_reward+(reward-average_reward)/env.t
				average_action = average_action+(action[0]-average_action)/env.t

				if env.t % 1000 == 0:
					env.render(mode=['console'])

				agent.buffer.record((prev_state, prev_res_state, action, reward, state, res_state))
				if agent.buffer.TD3:
					if env.t % 2:
						agent.buffer.learn_actor_critic(agent)
						update_target(agent.target_actor.variables, agent.actor.variables, tau)
						update_target(agent.target_critic.variables, agent.critic.variables, tau)	
						update_target(agent.target_critic2.variables, agent.critic2.variables, tau)					
					else:
						agent.buffer.learn_critic(agent)
				else:
					agent.buffer.learn(agent)
					update_target(agent.target_actor.variables, agent.actor.variables, tau)
					update_target(agent.target_critic.variables, agent.critic.variables, tau)	

				prev_state = state
				prev_res_state = res_state

			agent.epi_reward_list.append(episodic_reward)
			agent.avg_reward_list.append(average_reward)
			agent.avg_action_list.append(average_action)
			agent.epi_avg_reward_list.append(np.mean(agent.epi_reward_list[-40:]))

			if agent.epi_count % 10 == 0:
				# save the weights every n episodes
				agent.save_weights(agent.epi_count,STOR_DIR)
				env.render(agent=agent,STOR_DIR=STOR_DIR,mode=['console','figures'])

			print("Episode * {} * Avg Reward is ==> {}".format(agent.epi_count, agent.epi_avg_reward_list[-1]))
			if agent.epi_count >= max_epi:
				exit()

if __name__ == '__main__':
	main()
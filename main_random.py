
# libraries:
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr # not currently used
import tensorflow as tf

# environment and utils:
from folsom.folsom import FolsomEnv
from machinery import *
# from models import *
from agents import Random_Agent

# based on example at: https://keras.io/examples/rl/ddpg_pendulum/

def main():

	DATA_DIR = Path.cwd()	
	STOR_DIR = DATA_DIR / 'random_results'
	STOR_DIR.mkdir(exist_ok=True)

	num_res_states = 2
	inflow_stack = 0 # 23 future inflow features at various time scales between 1 day and 5 years
	epi_steps = 5 * 365

	env = FolsomEnv(DATA_DIR,res_dim=(num_res_states,),inflow_stack=inflow_stack,epi_length=epi_steps,ens='r1i1p1',) # policy: storage, inflows, day of year
	agent, env = Random_Agent.create(env,eps=0.1) # this has actor/critic and target networks contained within
	obs = env.reset(agent_type=env.agent_type,model='canesm2',ens='r1i1p1')

	# get the environment and extract the number of actions.
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

	# discount factor for future rewards
	gamma = 0.99

	batch_size = 365
	buffer_capacity = 50000 
	buffer = Buffer(num_states,num_res_states,num_actions=num_actions,buffer_capacity=buffer_capacity,batch_size=batch_size,gamma=gamma)

	"""
	Implement main training loop, and iterate over episodes.
	Sample actions using `policy()` and train with `learn()` at each time step.
	Update the Target networks at a rate `tau`.
	"""
	epi_reward_list = []
	avg_reward_list = []
	avg_action_list = []

	# ensembles = ['r{}i1p1'.format(i) for i in np.arange(1,2)]
	ensembles = ['r1i1p1' for i in np.arange(0,10)]
	results = {} # not used yet
	epi_count = 0

	for ens in ensembles:
		print('Running data from ensemble member: {}'.format(ens))
		ens_done = False
		average_reward = 0
		average_action = 0
		while ens_done == False:
			prev_state = env.reset(agent_type=env.agent_type,model='canesm2',ens=ens,epi_count=epi_count)
			prev_res_state = prev_state
			episodic_reward = 0
			episodic_action = 0
			epi_done = False
			epi_count += 1
			while epi_done == False:
				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
				action = agent.policy(ou_noise, env)
				# print(action[0]) # target release
				# Recieve state and reward from environment.
				state, reward, info = env.step(action)
				res_state = state
				ens_done, epi_done = info['ens_done'], info['epi_done']
				if env.t % 100 == 0:
					print('|| Ep: {} ||'.format('{:1.0f}'.format(epi_count)),
						't: {} ||'.format('{:5.0f}'.format(env.t)),
						'dowy: {} ||'.format('{:3.0f}'.format(env.dowy)),
						'R: {} ||'.format('{:7.0f}'.format(reward)),
						'Ep. R: {} ||'.format('{:8.0f}'.format(episodic_reward)),
						'Avg. R: {} ||'.format('{:4.0f}'.format(average_reward)),
						'S: {} ||'.format('{:3.0f}'.format(env.S[env.t])),
						'A: {} ||'.format('{:3.0f}'.format(env.action)), 
						'Avg. A: {} ||'.format('{:3.0f}'.format(average_action)),	
						'I: {} ||'.format('{:3.0f}'.format(env.Q[env.t])),							
						'O: {} ||'.format('{:3.0f}'.format(env.R[env.t])),
						)
				buffer.record((prev_state, prev_res_state, action, reward, state, res_state))
				episodic_reward += reward
				average_reward = average_reward+(reward-average_reward)/env.t
				average_action = average_action+(action[0]-average_action)/env.t

				# if env.t > batch_size*epi_count or epi_count*epi_steps > env.T: # don't learn until at least a full batch is in buffer
					# buffer.learn(agent)
					# update_target(agent.target_actor.variables, agent.actor.variables, tau)
					# update_target(agent.target_critic.variables, agent.critic.variables, tau)
				
				prev_state = state
				prev_res_state = res_state


			epi_reward_list.append(episodic_reward)
			avg_reward_list.append(average_reward)
			avg_action_list.append(average_action)
			print("Episode * {} * Avg Reward is ==> {}".format(epi_count, np.mean(epi_reward_list)))
			
			# save the weights every episode (this could be too frequent)
			# actor_model.save_weights(STOR_DIR / "actor_{}ep.h5".format(epi_count))
			# critic_model.save_weights(STOR_DIR / "critic_{}ep.h5".format(epi_count))

			# target_actor.save_weights(STOR_DIR / "target_actor_{}ep.h5".format(epi_count))
			# target_critic.save_weights(STOR_DIR / "target_critic_{}ep.h5".format(epi_count))

			# plot results and store DataFrame
			# plot_df = {'Episodic Rewards': epi_reward_list,
			# 			'Average Rewards': avg_reward_list,
			# 			'Average Actions': avg_action_list,}
			# plot_df=pd.DataFrame.from_dict(plot_df)
			# plot_df.to_csv(STOR_DIR / 'random_results.csv')
			# axes = plot_df.plot(subplots=True,figsize=(8,6))
			# axes[0].set_title('Random Agent - eps = {}'.format(env.eps))
			# axes[0].set_ylabel('Tot. Cost')
			# axes[1].set_ylabel('Avg. Cost')
			# axes[2].set_ylabel('Release [TAF]')
			# axes[2].set_xlabel('Episode')
			# for ax in axes.flatten():
			# 	ax.legend(frameon=False)
			# plt.tight_layout()
			# plt.savefig(STOR_DIR / 'random_results.png',dpi=400)

if __name__ == '__main__':
	main()
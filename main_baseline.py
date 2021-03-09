
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
from models import *

# based on example at: https://keras.io/examples/rl/ddpg_pendulum/

def main():
	DATA_DIR = Path.cwd()

	num_res_states = 2
	inf_stack = 1
	epi_steps = 10 * 365

	env = FolsomEnv(res_dim=(num_res_states,),inflow_stack=inf_stack, epi_length=epi_steps) # policy: storage, inflows, day of year
	obs = env.reset(DATA_DIR,model='canesm2',ens='r1i1p1')

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
	std_dev = 60. # this is high
	ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1)) # not currently used

	actor_model = get_actor(env)
	print(actor_model.summary())
	critic_model = get_critic(env)
	print(critic_model.summary())

	target_actor = get_actor(env)
	target_critic = get_critic(env)

	# making the weights equal initially:
	target_actor.set_weights(actor_model.get_weights())
	target_critic.set_weights(critic_model.get_weights())

	# learning rate for actor-critic models:
	actor_lr = 10e-6 # slowest learner
	critic_lr = 10e-5 # should learn faster than actor
	# learning rate used to update target networks:
	tau = 10e-4 # both actor and critic should chase the target actor and critic networks

	# initialize optimizers:
	clipnorm = 0.01
	# critic_optimizer = tf.keras.optimizers.SGD(critic_lr,clipnorm=clipnorm,momentum=0.5) # also sorta worked
	# actor_optimizer = tf.keras.optimizers.SGD(actor_lr,clipnorm=clipnorm,momentum=0.5)
	critic_opt = tf.keras.optimizers.Adagrad(critic_lr) 		#,clipnorm=clipnorm)
	actor_opt = tf.keras.optimizers.Adagrad(actor_lr)		#,clipnorm=clipnorm)
	# Adagrad emphasizes rarely seen experiences, keeps a running history of weight updates

	# discount factor for future rewards
	gamma = 0.99

	batch_size = 365
	buffer_capacity = 15000 
	buffer = Buffer(actor_opt,critic_opt,num_states,num_res_states,num_actions,buffer_capacity,batch_size,gamma)

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
			prev_state = env.reset(DATA_DIR,model='canesm2',ens=ens,epi_count=epi_count)
			prev_res_state = prev_state
			episodic_reward = 0
			episodic_action = 0
			epi_done = False
			epi_count += 1
			while epi_done == False:
				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
				action = policy(tf_prev_state, ou_noise, actor_model, env)
				# print(action[0]) # target release
				# Recieve state and reward from environment.
				state, reward, info = env.step(action)
				res_state = state
				ens_done, epi_done = info['ens_done'], info['epi_done']
				if env.t % 10 == 0:
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

				if env.t > batch_size*epi_count or epi_count*epi_steps > env.T: # don't learn until at least a full batch is in buffer
					target_actor,target_critic,actor_model,critic_model = buffer.learn(target_actor,target_critic,actor_model,critic_model)
					update_target(target_actor.variables, actor_model.variables, tau)
					update_target(target_critic.variables, critic_model.variables, tau)
				
				prev_state = state
				prev_res_state = res_state


			epi_reward_list.append(episodic_reward)
			avg_reward_list.append(average_reward)
			avg_action_list.append(average_action)
			print("Episode * {} * Avg Reward is ==> {}".format(epi_count, np.mean(epi_reward_list[-40:])))
			
			# save the weights every episode (this could be too frequent)
			# actor_model.save_weights(STOR_DIR / "actor_{}ep.h5".format(epi_count))
			# critic_model.save_weights(STOR_DIR / "critic_{}ep.h5".format(epi_count))

			# target_actor.save_weights(STOR_DIR / "target_actor_{}ep.h5".format(epi_count))
			# target_critic.save_weights(STOR_DIR / "target_critic_{}ep.h5".format(epi_count))

	# plot results and store DataFrame
	plot_df = {'Episodic Rewards': epi_reward_list,
				'Average Rewards': avg_reward_list,
				'Average Actions': avg_action_list,}
	plot_df=pd.DataFrame.from_dict(plot_df)
	plot_df.to_csv(STOR_DIR / 'trial_{}_results.csv'.format(trial))
	axes = plot_df.plot(subplots=True,figsize=(8,6))
	axes[0].set_title(trial)
	axes[0].set_ylabel('Tot. Cost')
	axes[1].set_ylabel('Avg. Cost')
	axes[2].set_ylabel('Release [TAF]')
	axes[2].set_xlabel('Episode')
	for ax in axes.flatten():
		ax.legend(frameon=False)
	plt.tight_layout()
	plt.savefig(STOR_DIR / 'trial_{}_results.png'.format(trial),dpi=300)

if __name__ == '__main__':
	main()
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential

"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor(env):
	# initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1)
	initializer = tf.random_uniform_initializer(minval=-0.3, maxval=0.3)
	last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)
	# state_input = layers.Input(shape=env.observation_space.shape)
	# state_out = layers.experimental.preprocessing.Rescaling(1./255)(state_input)
	# state_out = layers.Conv3D(8,5,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.Conv3D(8,4,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.AveragePooling3D()(state_out)
	# state_out = layers.Conv3D(8,3,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.AveragePooling3D()(state_out)
	# state_out = layers.Flatten()(state_out)

	res_input = layers.Input(shape=env.reservoir_space.shape)
	res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365])(res_input)
	res_out = layers.Dense(32,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(res_input)		
	res_out = layers.Dense(32,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)

	# concat here later

	out = layers.Dense(64,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)		
	out = layers.Dense(64,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(out)
	out = layers.Dense(1,activation='sigmoid',kernel_initializer=last_init,kernel_regularizer='l2')(out)
	out = out*env.action_space.high[0]
	model = tf.keras.Model(res_input,out)
	return model

def get_critic(env):

	initializer = tf.random_uniform_initializer(minval=-0.3, maxval=0.3)
	last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

	# state input - climate images
	# state_input = layers.Input(shape=env.observation_space.shape)
	# state_out = layers.experimental.preprocessing.Rescaling(1./255)(state_input)
	# state_out = layers.Conv3D(8,5,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.Conv3D(8,4,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.AveragePooling3D()(state_out)
	# state_out = layers.Conv3D(8,3,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
	# state_out = layers.AveragePooling3D()(state_out)
	# state_out = layers.Flatten()(state_out)

	# reservoir input
	res_input = layers.Input(shape=env.reservoir_space.shape)
	# res_out = layers.experimental.preprocessing.Normalization(dtype=tf.float32,mean=[])(res_input) # need this?
	res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365])(res_input)
	res_out = layers.Dense(32,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)		
	res_out = layers.Dense(32,activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)

	# action input
	act_input = layers.Input(shape=env.action_space.shape)
	act_out = layers.experimental.preprocessing.Rescaling(scale=1./env.action_space.high[0])(act_input) # need this?

	concat = layers.Concatenate()([act_out, res_out])

	out = layers.Dense(64,activation="relu",kernel_initializer=initializer,kernel_regularizer='l2')(concat)
	out = layers.Dense(64,activation="relu",kernel_initializer=initializer,kernel_regularizer='l2')(out)
	out = layers.Dense(1,activation="tanh",kernel_initializer=last_init,kernel_regularizer='l2')(out)

	# Outputs single value for give state-action
	model = tf.keras.Model([res_input,act_input], out)	

	return model

def policy(res_state, noise_object, actor_model, env):
	upper_bound = env.action_space.high[0]
	lower_bound = env.action_space.low[0]
	"""
	`policy()` returns an action sampled from our Actor network plus some noise for
	exploration.
	"""
	sampled_actions = tf.squeeze(actor_model(res_state))
	draw = np.random.uniform(0,1)
	# time = env.t/env.T
	if draw < 0.1:
		# noise = noise_object()
		# if sampled_actions.numpy() < 0.5*env.action_space.high[0]:
		# sampled_actions = sampled_actions.numpy() + noise				
		sampled_actions = sampled_actions.numpy() + np.random.uniform(-0.1*env.action_space.high[0],0.1*env.action_space.high[0])
		# else:
			# sampled_actions = sampled_actions.numpy() - np.random.uniform(0,0.5*env.action_space.high[0])
	else:
		sampled_actions = sampled_actions.numpy()		

	# We make sure action is within bounds
	legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

	return [np.squeeze(legal_action)]
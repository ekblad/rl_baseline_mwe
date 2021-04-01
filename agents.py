import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
import pandas as pd 
import numpy as np

class Random_Agent:
	"""
	This agent takes actions corresponding to the actual demand in (1-eps) percent of cases, and random actions in (eps) percent of cases.
	The agent need not learn in the training loop, and does not need action-selection networks, because the action-selection policy is deterministic.
	"""		
	def __init__(self, env, eps = 0.1):
		self.env = env
		self.eps = eps
		self.env.eps = self.eps
		self.agent_type = 'random'
		self.env.agent_type = 'random'

	@staticmethod
	def create(env,eps):
		obj = Random_Agent(env, eps)
		return obj, obj.env

	def policy(self, noise_object, env):
		"""
		`policy()` returns the demand as a target action in (1-eps) percent of cases, and random actions in (eps) percent of cases.
		"""
		upper_bound = env.action_space.high[0]
		lower_bound = env.action_space.low[0]
		sampled_actions = tf.squeeze(env.D[env.t])
		draw = np.random.uniform(0,1)
		if draw < self.eps:
			noise = noise_object()
			sampled_actions = sampled_actions.numpy() + noise
		else:
			noise = 0.
			sampled_actions = sampled_actions.numpy()		

		# We make sure action is within bounds
		legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

		return [np.squeeze(legal_action)], noise

class Planner:
	"""
	This agent has access to information about the future at different time scales, so it can plan releases around expected inflows.
	"""		
	def __init__(self, env, weights_dir = None, max_epi = 100, eps = None):
		self.env = env
		self.upper_bound = env.action_space.high[0]
		self.lower_bound = env.action_space.low[0]
		self.eps = eps
		self.env.eps = self.eps
		# self.env.Q_df = self.env.Q_df
		self.flows_format()
		self.Q_num_inputs = len(self.env.Q_future.columns)
		self.agent_type = 'planner'
		self.weights_dir = weights_dir
		self.max_epi = max_epi
		self.reset()

	@staticmethod
	def create(env, weights_dir = None, max_epi = 100, eps = None):
		obj = Planner(env, weights_dir = weights_dir, max_epi = max_epi, eps = eps)
		return obj, obj.env

	def flows_format(self):
		flows = self.env.Q_df.copy(deep=False)
		flows['inf_t+1'] = np.append(flows.inflow.values[1:],np.NaN)
		for i in range(2,6):
			flows['inf_t+{}'.format(i)] = np.append(flows['inf_t+{}'.format(i-1)].values[1:],np.NaN)
		for _,(i,j) in enumerate(zip(['5d','1m','3m','6m','1y','2y','3y','4y','5y'],[5,30,90,180,365,730,1095,1460,1825])):	
			flows['inf_{}_mean'.format(i)] = flows['inflow'].iloc[::-1].rolling(window='{}D'.format(j)).mean().iloc[::-1]
			# flows['inf_{}_sum'.format(i)] = flows['inflow'].iloc[::-1].rolling(window='{}D'.format(j)).sum().iloc[::-1]
		self.env.Q_future = flows.ffill()
		self.env.Q_future_numpy = self.env.Q_future.values

	def get_actor(self,env):

		# initializers:
		initializer = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
		last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

		# state_input = layers.Input(shape=env.observation_space.shape)
		# state_out = layers.experimental.preprocessing.Rescaling(1./255)(state_input)
		# state_out = layers.Conv3D(8,5,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.Conv3D(8,4,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.AveragePooling3D()(state_out)
		# state_out = layers.Conv3D(8,3,padding='same',activation='relu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.AveragePooling3D()(state_out)
		# state_out = layers.Flatten()(state_out)

		if self.agent_type == 'planner':
			res_input = layers.Input(shape=env.reservoir_space.shape)
			res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365]+list(1./env.Q_future.max().values))(res_input)
		# elif self.agent_type == 'scalar_climate':
			# do something
		# elif self.agent_type == 'hybrid_climate':
			# do something
		elif self.agent_type == 'baseline' or self.agent_type == 'spatial_climate':
			res_input = layers.Input(shape=env.reservoir_space.shape)
			res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365])(res_input)
		res_out = layers.Dense(128,activation='selu',kernel_initializer=initializer)(res_out)		
		res_out = tf.keras.layers.BatchNormalization()(res_out)
		res_out = layers.Dense(128,activation='selu',kernel_initializer=initializer)(res_out)
		res_out = tf.keras.layers.BatchNormalization()(res_out)

		out = layers.Dense(256,activation='selu',kernel_initializer=initializer)(res_out)
		out = tf.keras.layers.BatchNormalization()(out)
		out = layers.Dense(256,activation='selu',kernel_initializer=initializer)(out)
		out = tf.keras.layers.BatchNormalization()(out)
		out = layers.Dense(1,activation='sigmoid',kernel_initializer=last_init)(out)
		out = env.action_space.low[0] + out*(env.action_space.high[0] - env.action_space.low[0])
		model = tf.keras.Model(res_input,out)
		return model

	def get_critic(self,env):

		initializer = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
		last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

		# state input - climate images
		# state_input = layers.Input(shape=env.observation_space.shape)
		# state_out = layers.experimental.preprocessing.Rescaling(1./255)(state_input)
		# state_out = layers.Conv3D(8,5,padding='same',activation='selu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.Conv3D(8,4,padding='same',activation='selu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.AveragePooling3D()(state_out)
		# state_out = layers.Conv3D(8,3,padding='same',activation='selu',kernel_initializer=initializer,kernel_regularizer='l2')(state_out)
		# state_out = layers.AveragePooling3D()(state_out)
		# state_out = layers.Flatten()(state_out)

		# reservoir input
		if self.agent_type == 'planner':
			res_input = layers.Input(shape=env.reservoir_space.shape)
			res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365]+list(1./env.Q_future.max().values))(res_input)
		# elif self.agent_type == 'scalar_climate':
			# do something
		# elif self.agent_type == 'hybrid_climate':
			# do something
		elif self.agent_type == 'baseline' or self.agent_type == 'spatial_climate':
			res_input = layers.Input(shape=env.reservoir_space.shape)
			res_out = layers.experimental.preprocessing.Rescaling(scale=[1./env.K,1./365])(res_input)

		res_out = layers.Dense(128,activation='selu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)	
		res_out = tf.keras.layers.BatchNormalization()(res_out)	
		res_out = layers.Dense(128,activation='selu',kernel_initializer=initializer,kernel_regularizer='l2')(res_out)
		res_out = tf.keras.layers.BatchNormalization()(res_out)

		# action input
		act_input = layers.Input(shape=env.action_space.shape)
		act_out = layers.experimental.preprocessing.Rescaling(scale=1./env.action_space.high[0])(act_input)

		concat = layers.Concatenate()([act_out, res_out])

		out = layers.Dense(256,activation="selu",kernel_initializer=initializer,kernel_regularizer='l2')(concat)
		out = tf.keras.layers.BatchNormalization()(out)
		out = layers.Dense(256,activation="selu",kernel_initializer=initializer,kernel_regularizer='l2')(out)
		out = tf.keras.layers.BatchNormalization()(out)
		out = layers.Dense(1,activation="tanh",kernel_initializer=last_init,kernel_regularizer='l2')(out)

		# Outputs single value for give state-action
		model = tf.keras.Model([res_input,act_input], out)	

		return model

	def policy(self, res_state, noise_object, actor_model, epi_count):
		"""
		`policy()` returns an action sampled from our Actor network plus some noise for
		exploration.
		"""
		sampled_actions = tf.squeeze(actor_model(res_state))
		draw = np.random.uniform(0,1)
		if self.eps is None:
			if draw > epi_count/self.max_epi:
				noise = noise_object()
				sampled_actions = sampled_actions.numpy() + noise
			else:
				noise = 0.
				sampled_actions = sampled_actions.numpy()
		else:	
			if draw < self.eps:
				noise = noise_object()
				sampled_actions = sampled_actions.numpy() + noise
			else:
				noise = 0.
				sampled_actions = sampled_actions.numpy()
		# make sure action is within bounds
		legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
		return [np.squeeze(legal_action)], noise

	def reset(self):
		self.actor = self.get_actor(self.env)
		self.critic = self.get_critic(self.env)
		self.critic2 = self.get_critic(self.env)
		self.target_actor = self.get_actor(self.env)
		self.target_critic = self.get_critic(self.env)
		self.target_critic2 = self.get_critic(self.env)
		if self.weights_dir is not None:
			self.WEIGHTS_DIR = self.weights_dir / 'weights'
			# loading most recent saved weights
			self.actor.load_weights(sorted(self.WEIGHTS_DIR.glob('actor_*.h5'))[-1])
			self.critic.load_weights(sorted(self.WEIGHTS_DIR.glob('critic_*.h5'))[-1])
			self.critic2.load_weights(sorted(self.WEIGHTS_DIR.glob('critic2_*.h5'))[-1])
			self.target_actor.load_weights(sorted(self.WEIGHTS_DIR.glob('target_actor_*.h5'))[-1])
			self.target_critic.load_weights(sorted(self.WEIGHTS_DIR.glob('target_critic_*.h5'))[-1])
			self.target_critic2.load_weights(sorted(self.WEIGHTS_DIR.glob('target_critic2_*.h5'))[-1])
		else:
			# making the weights equal initially:
			self.target_actor.set_weights(self.actor.get_weights())
			self.target_critic.set_weights(self.critic.get_weights())
			self.target_critic2.set_weights(self.critic2.get_weights())

	def save_weights(self, epi_count, STOR_DIR):
		self.WEIGHTS_DIR = STOR_DIR / 'weights'
		if not self.WEIGHTS_DIR.exists():
			self.WEIGHTS_DIR.mkdir(exist_ok=True)
		self.actor.save_weights(self.WEIGHTS_DIR / f"actor_{epi_count:05}.h5")
		self.critic.save_weights(self.WEIGHTS_DIR / f"critic_{epi_count:05}.h5")
		self.critic2.save_weights(self.WEIGHTS_DIR / f"critic2_{epi_count:05}.h5")
		self.target_actor.save_weights(self.WEIGHTS_DIR / f"target_actor_{epi_count:05}.h5")
		self.target_critic.save_weights(self.WEIGHTS_DIR / f"target_critic_{epi_count:05}.h5")
		self.target_critic2.save_weights(self.WEIGHTS_DIR / f"target_critic2_{epi_count:05}.h5")



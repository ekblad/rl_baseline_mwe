import numpy as np
import tensorflow as tf

# based on example at: https://keras.io/examples/rl/ddpg_pendulum/

class OUActionNoise:
	"""
	To implement better exploration by the Actor network, we use noisy perturbations,
	specifically an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper. 
	It samples noise from a correlated normal distribution.
	"""		
	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		# formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
		x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
		# store x into x_prev - makes next noise dependent on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)

class Buffer:
	"""
	The `Buffer` class implements Experience Replay.
	**Critic loss** - Mean Squared Error of `y - Q(s, a)`
	where `y` is the expected return as seen by the Target network,
	and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
	that the critic model tries to achieve; we make this target
	stable by updating the Target model slowly.
	**Actor loss** - This is computed using the mean of the value given by the Critic network
	for the actions taken by the Actor network. We seek to maximize this quantity.
	Hence we update the Actor network so that it produces actions that get
	the maximum predicted value as seen by the Critic, for a given state.
	"""		
	def __init__(self, env, agent, buffer_capacity=100000, batch_size=64, gamma=0.99):
		# self.TD3 = agent.TD3
		# store optimizers
		self.actor_optimizer = agent.actor_optimizer
		self.critic_optimizer = agent.critic_optimizer
		self.critic2_optimizer = agent.critic2_optimizer
		self.TD3 = agent.TD3
		self.num_states = env.observation_space.shape
		self.num_actions = env.action_space.shape[0]
		self.num_res_states = env.reservoir_space.shape
		# number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# num of tuples to train on.
		self.batch_size = batch_size
		self.gamma = gamma

		# num of times record() was called.
		self.buffer_counter = 0

		# instead of list of tuples as the exp.replay concept go, we use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity,) + self.num_states)
		self.res_state_buffer = np.zeros((self.buffer_capacity,)+ self.num_res_states)			
		self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity,)+ self.num_states)
		self.next_res_state_buffer = np.zeros((self.buffer_capacity,)+ self.num_res_states)

	# takes (s,a,r,s') obervation tuple as input
	def record(self, obs_tuple):
		# set index to zero if buffer_capacity is exceeded, replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.res_state_buffer[index] = obs_tuple[1]			
		self.action_buffer[index] = obs_tuple[2]
		self.reward_buffer[index] = obs_tuple[3]
		self.next_state_buffer[index] = obs_tuple[4]
		self.next_res_state_buffer[index] = obs_tuple[5]

		self.buffer_counter += 1

	# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
	# TensorFlow to build a static graph out of the logic and computations in our function.
	# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
	@tf.function
	def update_critic(self,res_state_batch,action_batch,reward_batch,next_res_state_batch,agent):
		# training and updating Actor & Critic networks.
		if self.TD3:
			with tf.GradientTape(persistent=True) as tape:
				target_action_noise = tf.clip_by_value(tf.random.normal([next_res_state_batch.shape[0],1],mean=0.,stddev=1.),-3.,3.)
				target_actions = tf.clip_by_value(agent.target_actor(next_res_state_batch, training=True)+target_action_noise,agent.lower_bound,agent.upper_bound)
				target_critic = tf.reduce_min(tf.stack([agent.target_critic([next_res_state_batch,target_actions],training=True),agent.target_critic2([next_res_state_batch,target_actions],training=True)],axis=-1),axis=-1)
				y = reward_batch + self.gamma * target_critic
				critic_value = agent.critic([res_state_batch, action_batch], training=True)
				critic2_value = agent.critic2([res_state_batch, action_batch], training=True)
				critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
				critic2_loss = tf.math.reduce_mean(tf.math.square(y - critic2_value))
			critic_grad = tape.gradient(critic_loss, agent.critic.trainable_variables)
			critic2_grad = tape.gradient(critic2_loss, agent.critic2.trainable_variables)
			del tape
			self.critic_optimizer.apply_gradients(zip(critic_grad, agent.critic.trainable_variables))
			self.critic2_optimizer.apply_gradients(zip(critic2_grad, agent.critic2.trainable_variables))
		else:
			with tf.GradientTape() as tape:
				target_actions = agent.target_actor(next_res_state_batch, training=True)
				y = reward_batch + self.gamma * agent.target_critic([next_res_state_batch, target_actions], training=True)
				critic_value = agent.critic([res_state_batch, action_batch], training=True)
				critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value)) 
				# error between target_critic on next state and critic_model on current state
			critic_grad = tape.gradient(critic_loss, agent.critic.trainable_variables)
			self.critic_optimizer.apply_gradients(zip(critic_grad, agent.critic.trainable_variables))

	@tf.function		
	def update_actor(self,res_state_batch,agent):
		with tf.GradientTape() as tape:
			actions = agent.actor(res_state_batch, training=True)
			critic_value = agent.critic([res_state_batch, actions], training=True)
			# used `-value` as we want to maximize the value given by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value) # critic prediction is actor loss
		actor_grad = tape.gradient(actor_loss, agent.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_grad, agent.actor.trainable_variables))

	def learn_actor_critic(self,agent):
		# get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# convert to tensors
		# state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		res_state_batch = tf.convert_to_tensor(self.res_state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		# next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
		next_res_state_batch = tf.convert_to_tensor(self.next_res_state_buffer[batch_indices])
		self.update_critic(res_state_batch, action_batch, reward_batch, next_res_state_batch, agent)
		self.update_actor(res_state_batch, agent)

	def learn_critic(self,agent):
		# get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# convert to tensors
		# state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		res_state_batch = tf.convert_to_tensor(self.res_state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		# next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
		next_res_state_batch = tf.convert_to_tensor(self.next_res_state_buffer[batch_indices])
		self.update_critic(res_state_batch, action_batch, reward_batch, next_res_state_batch, agent)

	@tf.function
	def update(self, res_state_batch,action_batch,reward_batch,next_res_state_batch,agent):
		# training and updating Actor & Critic networks. - see pseudocode.
		with tf.GradientTape() as tape:
			target_actions = agent.target_actor(next_res_state_batch, training=True)
			y = reward_batch + self.gamma * agent.target_critic([next_res_state_batch, target_actions], training=True)
			critic_value = agent.critic([res_state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value)) 
			# error between target_critic on next state and critic_model on current state
		critic_grad = tape.gradient(critic_loss, agent.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grad, agent.critic.trainable_variables))

		with tf.GradientTape() as tape:
			actions = agent.actor(res_state_batch, training=True)
			critic_value = agent.critic([res_state_batch, actions], training=True)
			# used `-value` as we want to maximize the value given by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value) # critic prediction is actor loss
		actor_grad = tape.gradient(actor_loss, agent.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_grad, agent.actor.trainable_variables))

	# compute the loss and update parameters
	def learn(self,agent):
		# get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# convert to tensors
		# state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		res_state_batch = tf.convert_to_tensor(self.res_state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		# next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
		next_res_state_batch = tf.convert_to_tensor(self.next_res_state_buffer[batch_indices])

		self.update(res_state_batch, action_batch, reward_batch, next_res_state_batch,agent)

	def warmup(self,env,agent,OLD_DIR):
		warmup, warmups, epi_done = True, 0, False
		# if OLD_DIR is None:
		while warmup:
			prev_state = env.reset(agent)
			prev_res_state = prev_state
			while epi_done == False:
				action, noise = [np.random.uniform(env.action_space.low[0],env.action_space.high[0]),], 0
				state, reward, info = env.step(action, noise)
				res_state = state
				ens_done, epi_done = info['ens_done'], info['epi_done']
				self.record((prev_state, prev_res_state, action, reward, state, res_state))
				prev_state = state
				prev_res_state = res_state
			warmup = False
			if warmups < self.buffer_capacity / agent.epi_steps:
				warmups += 1			
				epi_done = False
				warmup = True
		# else:
		# 	while warmup:
		# 		prev_state = env.reset(agent)
		# 		prev_res_state = prev_state
		# 		while epi_done == False:
		# 			tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
		# 			action, noise = agent.policy(tf_prev_state)
		# 			state, reward, info = env.step(action, noise)
		# 			res_state = state
		# 			ens_done, epi_done = info['ens_done'], info['epi_done']
		# 			self.record((prev_state, prev_res_state, action, reward, state, res_state))
		# 			prev_state = state
		# 			prev_res_state = res_state
		# 		warmup = False
		# 		if warmups < self.buffer_capacity / agent.epi_steps:
		# 			warmups += 1			
		# 			epi_done = False
		# 			warmup = True

# this update target parameters slowly based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1. - tau))
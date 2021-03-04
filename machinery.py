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
	def __init__(self,actor_optimizer,critic_optimizer,num_states,num_res_states,
				num_actions=1,buffer_capacity=100000,batch_size=64,gamma=0.99):
		# store optimizers
		self.actor_optimizer = actor_optimizer
		self.critic_optimizer = critic_optimizer
		# number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# num of tuples to train on.
		self.batch_size = batch_size
		self.gamma = gamma

		# num of times record() was called.
		self.buffer_counter = 0

		# instead of list of tuples as the exp.replay concept go, we use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity,) + num_states)
		self.res_state_buffer = np.zeros((self.buffer_capacity,)+ num_res_states)			
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity,)+ num_states)
		self.next_res_state_buffer = np.zeros((self.buffer_capacity,)+ num_res_states)

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
	def update(self, res_state_batch,action_batch,reward_batch,next_res_state_batch,target_actor,target_critic,actor_model,critic_model):
		# training and updating Actor & Critic networks. - see pseudocode.
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_res_state_batch, training=True)
			y = reward_batch + self.gamma * target_critic([next_res_state_batch, target_actions], training=True)
			critic_value = critic_model([res_state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value)) 
			# error between target_critic on next state and critic_model on current state
		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

		with tf.GradientTape() as tape:
			actions = actor_model(res_state_batch, training=True)
			critic_value = critic_model([res_state_batch, actions], training=True)
			# used `-value` as we want to maximize the value given by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value) # critic prediction is actor loss
		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

	# compute the loss and update parameters
	def learn(self,target_actor,target_critic,actor_model,critic_model):
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

		self.update(res_state_batch, action_batch, reward_batch, next_res_state_batch,target_actor,target_critic,actor_model,critic_model)
		return target_actor,target_critic,actor_model,critic_model


# this update target parameters slowly based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1. - tau))
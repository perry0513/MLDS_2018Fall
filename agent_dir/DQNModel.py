import tensorflow as tf
import random

class DQNModel():
	def __init__(self, action_size):
		self.seed = 0
		self.action_size = action_size
	
	def build_model(self, name: str, dqn_duel=False):
		self.state = tf.placeholder(tf.float32, (None, 84, 84, 4))
		with tf.variable_scope(name):
			conv1 = tf.layers.conv2d(
				inputs = self.state,
				filters = 32,
				kernel_size = [8,8],
				strides = [4,4],
				padding = 'valid',
				data_format = 'channels_last',
				activation = tf.nn.relu,
				kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
				bias_initializer = tf.zeros_initializer()
			)
			conv2 = tf.layers.conv2d(
				inputs = conv1,
				filters = 64,
				kernel_size = [4,4],
				strides = [2,2],
				padding = 'valid',
				data_format = 'channels_last',
				activation = tf.nn.relu,
				kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
				bias_initializer = tf.zeros_initializer()
			)
			conv3 = tf.layers.conv2d(
				inputs = conv2,
				filters = 64,
				kernel_size = [3,3],
				strides = [1,1],
				padding = 'valid',
				data_format = 'channels_last',
				activation = tf.nn.relu,
				kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
				bias_initializer = tf.zeros_initializer()
			)
			flatten = tf.layers.Flatten()(conv3)
			dense1 = tf.layers.dense(
				inputs = flatten,
				units = 512,
				activation = tf.nn.relu,
				kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
				bias_initializer = tf.zeros_initializer()
			)
			if dqn_duel:
				y = tf.layers.dense(
					inputs = dense1,
					units = self.action_size + 1,
					activation = None,
					kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
					bias_initializer = tf.zeros_initializer()
				)
				self.Q = tf.keras.layers.Lambda(lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:], keepdims=True), output_shape=(self.action_size,))(y)
			else:
				self.Q = tf.layers.dense(
					inputs = dense1,
					units = self.action_size,
					activation = None,
					kernel_initializer = tf.keras.initializers.glorot_uniform(seed=self.seed),
					bias_initializer = tf.zeros_initializer()
				)
			self.scope = tf.get_variable_scope().name

	def act(self, state, test, epsilon):
		if test:
			act_values = tf.get_default_session().run(self.Q, feed_dict={self.state:np.expand_dims(state, axis=0)})
			return np.argmax(act_values[0])
		else:
			if np.random.rand() <= epsilon:
				return random.randrange(self.action_size)
			act_values = tf.get_default_session().run(self.Q, {self.state:np.expand_dims(state, axis=0)})
			return np.argmax(act_values[0])

	def Q(self, state):
		return tf.get_default_session().run(self.Q, {self.state:np.expand_dims(state, axis=0)})

	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

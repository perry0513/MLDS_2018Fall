import tensorflow as tf

class PPOModel():
	def __init__(self, name, action_size):
		self.seed = 0
		self.action_size = action_size
		with tf.variable_scope(name):
			self.states = tf.placeholder(tf.float32, [None, 80, 80, 1])
			with tf.variable_scope('policy_net', reuse=False):
				conv1 = tf.layers.conv2d(
						inputs=self.states,
						filters=32,
						kernel_size=[8, 8],
						strides=[4, 4],
						padding='same',
						data_format='channels_last',
						activation=tf.nn.relu,
						kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
						bias_initializer=tf.zeros_initializer())
				conv2 = tf.layers.conv2d(
						inputs=conv1,
						filters=64,
						kernel_size=[4, 4],
						strides=[2, 2],
						padding='same',
						data_format='channels_last',
						activation=tf.nn.relu,
						kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
						bias_initializer=tf.zeros_initializer())

				flatten = tf.layers.Flatten()(conv2)
				
				self.v_preds = tf.layers.dense(
						inputs=flatten,
						units=1,
						activation=None,
						kernel_initializer=tf.keras.initializers.he_uniform(seed=self.seed),
						bias_initializer=tf.zeros_initializer())

				dense1 = tf.layers.dense(
						inputs=flatten,
						units=128,
						activation=tf.nn.relu,
						kernel_initializer=tf.keras.initializers.he_uniform(seed=self.seed),
						bias_initializer=tf.zeros_initializer())

				self.act_probs = tf.layers.dense(
						inputs=dense1, 
						units=action_size, 
						activation=tf.nn.softmax,
						kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
						bias_initializer=tf.zeros_initializer())
			self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
			self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

			self.act_deterministic = tf.argmax(self.act_probs, axis=1)
		
	def act(self, states, stochastic=True):
		if stochastic:
			return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.states: states})
		else:
			return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.states: states})
	
	def get_action_prob(self, states):
		return tf.get_default_session().run(self.act_probs, feed_dict={self.states: states})
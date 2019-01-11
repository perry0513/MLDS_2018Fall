import tensorflow as tf

class PPOTTrain:
	def __init__(self, Policy, Old_Policy, gamma, clip_value=0.2, c_1=1, c_2=0.01):
		self.Policy = Policy
		self.Old_Policy = Old_Policy
		self.gamma = gamma

		# inputs for train_op
		with tf.variable_scope('train_inp'):
			self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
			self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
			self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
			self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
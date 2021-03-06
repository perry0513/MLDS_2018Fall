import tensorflow as tf
import numpy as np
import copy

class PPOTrain:
	def __init__(self, Policy, Old_Policy, gamma, clip_value=0.2, c_1=1, c_2=0.01):
		self.Policy = Policy
		self.Old_Policy = Old_Policy
		self.gamma = gamma

		pi_trainable = self.Policy.get_trainable_variables()
		old_pi_trainable = self.Old_Policy.get_trainable_variables()

		# assign_operations for policy parameter values to old policy parameters
		with tf.variable_scope('assign_op'):
			self.assign_ops = []
			for v_old, v in zip(old_pi_trainable, pi_trainable):
				self.assign_ops.append(tf.assign(v_old, v))

		# inputs for train_op
		with tf.variable_scope('train_inp'):
			self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
			self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
			self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
			self.adv_func = tf.placeholder(dtype=tf.float32, shape=[None], name='adv_func')
		
		act_probs = self.Policy.act_probs
		act_probs_old = self.Old_Policy.act_probs

		# probabilities of actions which agent took with policy
		act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
		act_probs = tf.reduce_sum(act_probs, axis=1)

		# probabilities of actions which agent took with old policy
		act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
		act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

		with tf.variable_scope('loss'):
			# construct computation graph for loss_clip
			ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
							- tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
			clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
			loss_clip = tf.minimum(tf.multiply(self.adv_func, ratios), tf.multiply(self.adv_func, clipped_ratios))
			loss_clip = tf.reduce_mean(loss_clip)

			# construct computation graph for loss of entropy bonus
			entropy = -tf.reduce_sum(self.Policy.act_probs *
									 tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
			entropy = tf.reduce_mean(entropy, axis=0)

			# construct computation graph for loss of value function
			v_preds = self.Policy.v_preds
			loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
			loss_vf = tf.reduce_mean(loss_vf)
			
			'''loss for gae'''
#			loss = loss_clip - c_1 * loss_vf + c_2 * entropy
#			loss = -loss
			
			'''loss for g_t'''
			loss = -loss_clip
		optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
		self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
		self.train_op = optimizer.minimize(loss, var_list=pi_trainable)
	
	def train(self, states, actions, adv_func, rewards, v_preds_next):
		tf.get_default_session().run(self.train_op, feed_dict={	self.Policy.states: states,
																self.Old_Policy.states: states,
																self.actions: actions,
																self.rewards: rewards,
																self.v_preds_next: v_preds_next,
																self.adv_func: adv_func})

	def assign_policy_parameters(self):
		return tf.get_default_session().run(self.assign_ops)
	
	def get_g_t(self, rewards):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, len(rewards))):
			running_add = running_add * self.gamma + rewards[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs
	
	def get_gaes(self, rewards, v_preds, v_preds_next):
		deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
		return gaes

	def get_grad(self, states, actions, adv_func, rewards, v_preds_next):
		return tf.get_default_session().run(self.gradients, feed_dict={	self.Policy.states: states,
																		self.Old_Policy.states: states,
																		self.actions: actions,
																		self.rewards: rewards,
																		self.v_preds_next: v_preds_next,
																		self.adv_func: adv_func})

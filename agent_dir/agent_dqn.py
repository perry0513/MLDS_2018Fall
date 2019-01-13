from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import os

class Agent_DQN(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_DQN,self).__init__(env)

		self.sess = tf.InteractiveSession()

		self.double_dqn = not args.without_double_dqn
		self.duel_dqn = not args.without_duel_dqn

		self.recent_avg_rewards = []
		self.memory = deque(maxlen=10000)

		self.save_history_period = args.save_history_period
		self.episodes = args.episodes
		self.learning_rate = args.learning_rate
		self.batch_size = 32

		self.action_size = self.env.action_space.n

		# hyper parameters
		self.gamma = 0.95
		self.epsilon = 1.0
		self.eplison_min = 0.025
		self.eplison_step = 100000
		self.epsilon_decay = (self.epsilon-self.eplison_min) / self.eplison_step

		self.model = DQNModel('model', self.action_size, self.duel_dqn)
		self.target_model = DQNModel('target_model', self.action_size, self.duel_dqn)

		self.checkpoints_dir = './checkpoints'
		self.checkpoint_file = os.path.join(self.checkpoints_dir, 'dqn.ckpt')

		self.model_weights = self.model.get_trainable_variables()
		self.target_model_weights = self.target_model.get_trainable_variables()

		with tf.variables_scope('assign_op'):
			self.assign_ops = []
			for model_w, target_w in zip(self.model_weights, self.target_model_weights):
				self.assign_ops.append(tf.assign(target_w, model_w))
		
		self.target = tf.placeholder(tf.float32, (None, self.action_size), name='target')
		self.loss = tf.reduce_mean((tf.keras.losses.mean_squared_error(self.target, self.model.Q)))
		self.train_op = tf.keras.optimizers.RMSprop(lr=self.learning_rate).get_updates(self.loss, self.model_weights)

		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.assign_ops)

		if args.test_dqn:
			#you can load your model here
			print('Loading trained model . . .')
			self.saver.restore(self.sess, self.checkpoint_file)


	def init_game_setting(self):
		"""

		Testing function will call this function at the begining of new game
		Put anything you want to initialize if necessary

		"""
		##################
		# YOUR CODE HERE #
		##################
		pass

	def replay(self, batch_size):
		batch = random.sample(self.memory, batch_size)

		states, actions, rewards, next_states, dones = [], [], [], [], []

		for state, action, reward, next_state, done in batch:
			states.append(state)
			actions.append(action)
			rewards.append(reward)
			next_states.append(next_state)
			dones.append(done)

		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)

		targets = self.sess.run(self.model.Q, feed_dict={self.model.state: states})
		next_actions = self.sess.run(self.model.Q, feed_dict={self.model.state: next_states})
		target_actions = self.sess.run(self.target_model.Q, feed_dict={self.target_model.state: next_states})

		if not self.double_dqn:
			targets[:, actions] = rewards + (1 - dones) * self.gamma * np.max(target_actions, axis=1)
		else:
			targets[:, actions] = rewards + (1 - dones) * self.gamma * target_actions[:, np.argmax(next_actions, axis=1)]



	def train(self):
		"""
		Implement your training algorithm here
		"""
		
		recent_episode_num = 100
		recent_rewards = []
		recent_avg_reward = None
		best_avg_reward = 0.0

		for num_episode in range(self.episodes):
			state = self.env.reset()
			done = False
			num_action = 0
			sum_reward = 0

			while not done:
				if self.epsilon > self.epsilon_min:
					self.epsilon -= self.epsilon_decay
				action = self.model.act(state, False, self.epsilon)
				next_state, reward, done, info = self.env.step(action)
				sum_reward += reward
				self.memory.append((state, action, reward, next_state, done))
				state = next_state
				if num_episode % 4 == 0:
					# TODO: update model
				if num_episode % 1000 == 0:
					self.sess.run(self.assign_ops)

				num_action += 1

			recent_rewards.append(sum_reward)
			if len(recent_rewards) > recent_episode_num:
				recent_rewards.pop(0)
			recent_avg_reward = sum(recent_rewards)/len(recent_rewards)
			self.recent_avg_rewards.append(recent_avg_reward)
			# TODO: print something out

			if recent_avg_reward > best_avg_reward:
				print ('[Save Checkpoint] Avg. reward improved from {:2.6f} to {:2.6f}'.format(
					best_avg_reward, recent_avg_reward))
				best_avg_reward = recent_avg_reward
				print ('Saving Checkpoint...')
				self.saver.save(self.sess, self.checkpoint_file)

			if episode % self.save_history_period == 0:
				np.save('recent_avg_rewards.npy', np.array(self.recent_avg_rewards))


	def make_action(self, observation, test=True):
		"""
		Return predicted action of your agent

		Input:
			observation: np.array
				stack 4 last preprocessed frames, shape: (84, 84, 4)

		Return:
			action: int
				the predicted action from trained model
		"""
		##################
		# YOUR CODE HERE #
		##################
		return self.act(observation, True)


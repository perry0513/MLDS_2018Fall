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

		if args.test_dqn:
			#you can load your model here
			print('loading trained model')

		self.recent_avg_rewards = []
		self.memory = deque(maxlen=10000)

		self.save_history_period = args.save_history_period
		self.episodes = args.episodes
		self.learning_rate = args.learning_rate

		self.action_size = self.env.action_space.n

		# hyper parameters
		self.gamma = 0.95
		self.epsilon = 1.0
		self.eplison_min = 0.05
		self.eplison_step = 100000
		self.epsilon_decay = (self.epsilon-self.eplison_min) / self.eplison_step

        self.model = DQNModel('model', self.action_size, True)
        self.target_model = DQNModel('target_model', self.action_size, True)

		self.checkpoints_dir = './checkpoints'
		self.checkpoint_file = os.path.join(self.checkpoints_dir, 'dqn.ckpt')

		self.model_weights = self.model.get_trainable_variables()
		self.target_model_weights = self.target_model.get_trainable_variables()

        with tf.variables_scope('assign_op'):
            self.assign_ops = []
            for model_w, target_w in zip(self.model_weights, self.target_model_weights):
                self.assign_ops.append(tf.assign(target_w, model_w))


	def init_game_setting(self):
		"""

		Testing function will call this function at the begining of new game
		Put anything you want to initialize if necessary

		"""
		##################
		# YOUR CODE HERE #
		##################
		pass


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
					tf.sess.run(self.assign_ops)

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


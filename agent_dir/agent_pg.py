from agent_dir.agent import Agent
import scipy
import numpy as np

import tensorflow as tf
import os
from agent_dir.PPOTrain import PPOTrain
from agent_dir.PPOModel import PPOModel

def prepro(o,image_size=[80,80]):
	"""
	Call this function to preprocess RGB image to grayscale image if necessary
	This preprocessing code is from
		https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
	
	Input: 
	RGB image: np.array
		RGB screen of game, shape: (210, 160, 3)
	Default return: np.array 
		Grayscale image, shape: (80, 80, 1)
	
	"""
	y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
	y = y.astype(np.uint8)
	resized = scipy.misc.imresize(y, image_size)
	return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_PG,self).__init__(env)

		self.sess = tf.InteractiveSession()

		if args.test_pg:
			#you can load your model here
			print('loading trained model')
		
		# Load Arguments
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.episodes = args.episodes
		self.discount_factor = args.discount_factor
		self.render = args.render
		
		# Saving data
		self.save_history_period = args.save_history_period
		self.checkpoints_dir = './checkpoints_pg'
		self.checkpoint_file = os.path.join(self.checkpoints_dir, 'policy_network.ckpt')
		
		# Testing
		self.action_size = 2 # up/down
		self.states = []
		self.actions = []
		self.rewards = []
		self.probs = []
		self.v_preds = []
		self.recent_avg_rewards = []

		self.env = env
		self.env.seed(2000)

		self.theta = PPOModel('theta', self.action_size)
		self.theta_k = PPOModel('theta_k', self.action_size)
		
		self.PPO = PPOTrain(self.theta, self.theta_k, gamma=self.discount_factor)
		
		self.saver = tf.train.Saver()

		if args.load_checkpoint:
			self.network.load_checkpoint()
			self.recent_avg_rewards = np.load('./recent_avg_rewards.npy').tolist()

		if args.test_pg:
			print('Loading trained model...')
			self.load_checkpoint()


	def load_checkpoint(self):
		print("Loading checkpoint...")
		self.saver.restore(self.sess, self.checkpoint_file)


	def save_checkpoint(self):
		print("Saving checkpoint...")
		self.saver.save(self.sess, self.checkpoint_file)

	def init_game_setting(self):
		"""

		Testing functi:qon will call this function at the begining of new game
		Put anything you want to initialize if necessary
		
		"""
		self.last_state = None
		self.recent_episode_num = 30
		self.recent_rewards = []
		self.recent_avg_reward = None
		self.best_avg_reward = -30.0


	def train(self):
		"""
		Implement your training algorithm here
		"""
		self.init_game_setting()

		self.sess.run(tf.global_variables_initializer())
		
		num_episode = 1
		while True:
			done = False
			sum_reward_per_episode = 0

			last_state = prepro(self.env.reset())
			action = self.env.action_space.sample()
			observation, reward, done, info = self.env.step(action)
			state = prepro(observation)

			num_rounds = 1
			num_actions = 1
			num_win = 0
			num_lose = 0

			while not done:
				if self.render:
					self.env.env.render()

				delta_state = state - last_state
				last_state = state

				action, v_pred = self.theta.act(states=np.expand_dims(delta_state, axis=0), stochastic=True)
				action = np.asscalar(action)
				v_pred = np.asscalar(v_pred)

				observation, reward, done, info = self.env.step(action+2) # 2 for up and 3 for down

				self.states.append(delta_state)
				self.actions.append(action)
				self.v_preds.append(v_pred)
				self.rewards.append(reward)

				state = prepro(observation)
				sum_reward_per_episode += reward
				num_actions += 1

				if reward == -1:
					num_lose += 1
				if reward == +1:
					num_win += 1
				if reward != 0:
					print ('Round [{:2d}] {:2d} : {:2d}'.format(num_rounds, num_lose, num_win), end='\r')
					num_rounds += 1
			
			v_preds_next = self.v_preds[1:] + [0] # next state of terminate state has 0 state value

			self.recent_rewards.append(sum_reward_per_episode)
			if len(self.recent_rewards) > self.recent_episode_num:
				self.recent_rewards.pop(0)

			recent_avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
			self.recent_avg_rewards.append(recent_avg_reward)

			print ('Episode {:d} | Actions {:4d} | Reward {:2.3f} | Avg. reward {:2.6f}'.format(num_episode, num_actions, sum_reward_per_episode, recent_avg_reward))
			
			if recent_avg_reward > self.best_avg_reward:
				print ('[Save Checkpoint] Avg. reward improved from {:2.6f} to {:2.6f}'.format(
					self.best_avg_reward, recent_avg_reward))
				self.best_avg_reward = recent_avg_reward
				self.save_checkpoint()
			
			# gaes denotes for generalized advantage estimations
			gaes = self.PPO.get_gaes(rewards=self.rewards, v_preds=self.v_preds, v_preds_next=v_preds_next)
			# g_t
			g_t = self.PPO.get_g_t(rewards=self.rewards)

			states = np.array(self.states).astype(dtype=np.float32)
			actions = np.array(self.actions).astype(dtype=np.int32)
			rewards = np.array(self.rewards).astype(dtype=np.float32)
			v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
			gaes = np.array(gaes).astype(dtype=np.float32)
			gaes = (gaes - gaes.mean()) / gaes.std()

			self.PPO.assign_policy_parameters()

			self.PPO.train(states=states, actions=actions, adv_func=g_t, rewards=rewards, v_preds_next=v_preds_next)
			
			self.states, self.actions, self.v_preds, self.rewards = [], [], [], []

			if num_episode % self.save_history_period == 0:
				np.save('recent_avg_rewards_pg.npy', np.array(self.recent_avg_rewards))
			num_episode += 1


	def make_action(self, observation, test=True):
		"""
		Return predicted action of your agent

		Input:
			observation: np.array
				current RGB screen of game, shape: (210, 160, 3)

		Return:
			action: int
				the predicted action from trained model
		"""
		if self.render:
			self.env.env.render()
		state = prepro(observation)
		delta_state = state if self.last_state is None else state - self.last_state
		self.last_state = state
		
		action, v_pred = self.theta.act(states=np.expand_dims(delta_state, axis=0), stochastic=False)
		action = np.asscalar(action)
		
		return action + 2



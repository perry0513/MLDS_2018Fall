from agent_dir.agent import Agent
import scipy
import numpy as np

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
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
        
        # Load Arguments
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.episodes = args.episodes
        self.save_history_period = args.save_history_period
        self.discount_factor = args.discount_factor
        self.render = args.render
        
        # 
        self.action_size = 2 # up/down
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.v_preds = []
        self.recent_avg_rewards = []

        self.env = env
        self.env.seed(2000)

        self.theta = PPOModel(self.action_size)
        self.theta_k = PPOModel(self.action_size)

        self.PPO = PPOTrain(self.theta, self.theta_k, gamma=self.discount_factor)



    def init_game_setting(self):
        """

        Testing functi:qon will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.recent_episode_num = 30
        self.recent_rewards = []
        self.recent_avg_reward = None
        self.best_avg_reward = -30.0
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        self.init_game_setting()
        
        for num_episode in range(self.episodes):
            done = False
            sum_reward_per_episode = 0

            last_state = prepro(self.env.reset())
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            state = prepro(observation)

            num_rounds = 1
            num_actions = 1
            num_wins = 0
            num_lose = 0

            while not done:
                if self.render:
                    self.env.render()

                delta_state = state - last_state
                last_state = state

                action, v_pred = self.theta.act(states=np.expand_dims(delta_state, axis=0), stochastic=True)
                
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

            recent_rewards.append(sum_reward_episode)
            if len(recent_rewards) > recent_episode_num:
                recent_rewards.pop(0)

            recent_avg_reward = sum(recent_rewards) / len(recent_rewards)
            self.recent_avg_rewards.append(recent_avg_reward)

            print ('Episode {:d} | Actions {:4d} | Reward {:2.3f} | Avg. reward {:2.6f}'.format(num_episode, num_actions, sum_reward_episode, recent_avg_reward))
            
            if recent_avg_reward > best_avg_reward:
                best_avg_reward = recent_avg_reward
                # TODO: save checkpoint
            
            # TODO: train
            gen_adv_ests = self.PPO.getGAEs(rewards=self.rewards, v_preds=self.v_preds, v_preds_next=self.v_preds_next)

            states = np.array(self.states).astype(dtype=np.float32)
            actions = np.array(self.actions).astype(dtype=np.int32)
            rewards = np.array(self.rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gen_adv_ests = np.array(gen_adv_ests).astype(dtype=np.float32)
            gen_adv_ests = (gen_adv_ests - gen_adv_ests.mean()) / gen_adv_ests.std()

            self.PPO.assign_policy_parameters()

            self.PPO.train(states=states, actions=actions, gen_adv_ests=gen_adv_ests, rewards=rewards, v_preds_next=v_preds_next)

        if num_episode % self.save_history_period == 0:
            np.save('recent_avg_reward.py', np.array(self.recent_avg_reward))


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
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()


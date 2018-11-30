import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import data_processor
from tqdm import tqdm
import os

class WGAN():
	def __init__(self, noise_dim):
		self.pic_path = './pics/'
		if not os.path.exists(self.pic_path):
			os.makedirs(self.pic_path)
		self.noise_dim = noise_dim
		self.clip_w = 0.01
		self.alpha = 0.1
		self.learning_rate = 0.0005
		self.display_step = 1
		plt.switch_backend('agg')
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=50)

	def discriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			# inputs: (batch_size, img_h, img_w, channel)
			conv1 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(7,7)), alpha=self.alpha)
			conv2 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv1 , filters=128, kernel_size=(7,7)), alpha=self.alpha)
			conv3 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=(7,7)), alpha=self.alpha)
			conv4 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv3, filters=512, kernel_size=(7,7)), alpha=self.alpha)
			flatten = tf.layers.flatten(conv4)
			# output has no activation function
			output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs, reuse=False):
		# inputs: (batch_size, noise_dim)
		with tf.variable_scope('generator', reuse=reuse):
			dense1 = tf.layers.dense(inputs, 384*4*4, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
			reshape = tf.reshape(dense1, shape=[-1, 4, 4, 384])
			conv_trans1 = tf.nn.relu(tf.layers.conv2d_transpose(inputs=reshape, filters=256, kernel_size=(5,5), strides=(2,2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))
			conv_trans2 = tf.nn.relu(tf.layers.conv2d_transpose(inputs=conv_trans1, filters=128, kernel_size=(5,5), strides=(2,2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))
			conv_trans3 = tf.nn.relu(tf.layers.conv2d_transpose(inputs=conv_trans2, filters=64, kernel_size=(5,5), strides=(2,2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))
			conv_trans4 = tf.nn.relu(tf.layers.conv2d_transpose(inputs=conv_trans3, filters=3, kernel_size=(5,5), strides=(2,2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)))
			output = tf.nn.tanh(conv_trans4)
		return output
# 		with tf.variable_scope('generator', reuse=reuse)
# 			dense1 = tf.nn.leaky_relu(tf.layers.dense(inputs, 128*16*16))
# 			reshape = tf.reshape(dense1, shape=[-1, 16, 16, 128])
# 			conv_trans1 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(inputs=reshape, filters=128, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
# 			conv_trans2 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(inputs=conv_trans1, filters=64, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
# 			conv3 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv_trans2, filters=3, kernel_size=(4,4), strides=(1,1), padding='same'), alpha=self.alpha)
# 			output = tf.nn.tanh(conv3)
# 		return output

	def build_model(self):
		self.g_inputs = tf.placeholder(shape=(None, self.noise_dim), dtype=tf.float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32, name="discriminator_inputs")

		self.g_sample = self.generator(self.g_inputs, tf.AUTO_REUSE)
		self.d_real = self.discriminator(self.d_inputs, tf.AUTO_REUSE)
		self.d_fake = self.discriminator(self.g_sample, tf.AUTO_REUSE)
		self.g_loss = -tf.reduce_mean(self.d_fake)
		self.w_distance = tf.reduce_mean(self.d_real) + self.g_loss
		
# 		self.d_loss = tf.reduce_mean( self.discriminator(self.d_inputs, tf.AUTO_REUSE) )
# 		self.g_loss = -tf.reduce_mean( self.discriminator(self.generator(self.g_inputs, False), tf.AUTO_REUSE) )
# 		self.w_distance =  self.d_loss + self.g_loss

		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

		self.d_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize( -self.w_distance, var_list=d_vars )
		self.g_optim = tf.train.AdamOptimizer().minimize( self.g_loss, var_list=g_vars )

		self.clip_D = [ var.assign(tf.clip_by_value(var, -self.clip_w, self.clip_w)) for var in d_vars ]
# 		self.fake_img = self.generator(self.g_inputs, reuse=tf.AUTO_REUSE)

	def train(self, start_epoch, epochs, batch_size, g_iter, d_iter, model_dir):
		dp = data_processor.DataProcessor()
		sample_noise = np.random.normal(0, 1, (5*5, self.noise_dim))

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			end_epoch = start_epoch + epochs
			for epoch in range(start_epoch, end_epoch):
				w_distance, g_loss = 0,0
				batched_img = dp.get_batch(batch_size, d_iter)
				
				for b, img in enumerate(batched_img):
					for it in range(d_iter):
						z = np.random.normal(0, 1, (batch_size, self.noise_dim))
						_, _, w_distance, d_real, d_fake = sess.run([ self.d_optim, self.clip_D, self.w_distance, tf.reduce_mean(self.d_real), tf.reduce_mean(self.d_fake) ], 
													  feed_dict={ self.g_inputs:z, self.d_inputs:img })
# 						print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | W_dist: {:.6f}'
# 							  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, d_iter, w_distance), end='\r')
					
					for it in range(g_iter):
						z = np.random.normal(0, 1, (batch_size, self.noise_dim))
						_, g_loss = sess.run([ self.g_optim, self.g_loss ], feed_dict={ self.g_inputs:z })
						
# 						print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | G_loss: {:.6f}'
# 							  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, g_iter, g_loss), end='\r')

					print('Epoch [{:>2}/{:>2}] | W_dist: {:.6f} | G_loss: {:.6f} | d_real: {:.6f} | d_fake: {:.6f}'
						  .format(epoch+1, end_epoch, w_distance, g_loss, d_real, d_fake) )

				if (epoch+1) % self.display_step == 0:
					samples = sess.run(self.g_sample, feed_dict={ self.g_inputs:sample_noise })
					samples = samples / 2 + 0.5
# 					print(samples[0])
					fig = self.visualize_result(samples)
					plt.savefig(self.pic_path+'{}.png'.format(str(epoch+1).zfill(4)), bbox_inches='tight')
					plt.close(fig)
					
			self.saver.save(sess, './model/model_'+str(end_epoch)+'/model_'+str(end_epoch))
			
	def infer(self, model_dir):
		sample_noise = np.random.normal(0, 1, (5*5, self.noise_dim))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			samples = sess.run(self.g_sample, feed_dict={ self.g_inputs:sample_noise })
			samples = samples / 2 + 0.5
			fig = self.visualize_result(samples)
			plt.savefig('./infer.png', bbox_inches='tight')
			plt.close(fig)
			
		
	def visualize_result(self, samples):
		fig = plt.figure(figsize=(5,5))
		gs = gridspec.GridSpec(5,5)
		gs.update(wspace=0.1, hspace=0.1)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')

			plt.imshow(sample)

		return fig


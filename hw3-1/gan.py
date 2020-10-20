import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import data_processor
from tqdm import tqdm
import os

class GAN():
	def __init__(self, noise_dim):
		self.pic_path = './gan_pics/'
		if not os.path.exists(self.pic_path):
			os.makedirs(self.pic_path)
		self.noise_dim = noise_dim
		self.alpha = 0.2
		self.learning_rate = 0.0002
		self.beta1 = 0.5
		self.display_step = 1
		# plt.switch_backend('agg')
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=50)

	def discriminator(self, inputs, reuse=False, is_training=True):
		with tf.variable_scope('discriminator', reuse=reuse):
			# inputs: (batch_size, img_h, img_w, channel)
			flatten1 = tf.layers.flatten(inputs)
			dense1 = tf.nn.leaky_relu(tf.layers.dense(flatten1,64), alpha=self.alpha)
			dense2 = tf.nn.leaky_relu(tf.layers.dense(dense1,128), alpha=self.alpha)
			dense3 = tf.nn.leaky_relu(tf.layers.dense(dense2,256), alpha=self.alpha)
			dense4 = tf.nn.leaky_relu(tf.layers.dense(dense3,512), alpha=self.alpha)
			flatten2 = tf.layers.flatten(dense4)
			# output has no activation function
			output = tf.layers.dense(flatten2,1)
		return output

	def generator(self, inputs, reuse=False, is_training=True):
		# inputs: (batch_size, noise_dim)
		with tf.variable_scope('generator', reuse=reuse):
			dense1 = tf.nn.leaky_relu(tf.layers.dense(inputs, 512), alpha=self.alpha)
			dense2 = tf.nn.leaky_relu(tf.layers.dense(dense1, 1024), alpha=self.alpha)
			dense3 = tf.nn.leaky_relu(tf.layers.dense(dense2, 2048), alpha=self.alpha)
			dense4 = tf.nn.leaky_relu(tf.layers.dense(dense3, 3*64*64), alpha=self.alpha)
			reshape = tf.reshape(dense4, [-1, 64, 64, 3])
			output = tf.nn.tanh(reshape)
		return output

	def build_model(self):
		self.g_inputs = tf.placeholder(shape=(None, self.noise_dim), dtype=tf.float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32, name="discriminator_inputs")

		self.g_sample = self.generator(self.g_inputs, tf.AUTO_REUSE, is_training=True)
		self.d_real = self.discriminator(self.d_inputs, tf.AUTO_REUSE, is_training=True)
		self.d_fake = self.discriminator(self.g_sample, tf.AUTO_REUSE, is_training=True)
		self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real)))
		self.d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))
		self.d_loss = self.d_real_loss + self.d_fake_loss
		
		self.g_infer = self.generator(self.g_inputs, tf.AUTO_REUSE, is_training=False)

		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

		self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.d_loss, var_list=d_vars )
		self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.g_loss, var_list=g_vars )

	def get_noise(self, batch_size):
# 		return np.random.normal(0, 1, (batch_size, self.noise_dim))
		return np.random.uniform(-0.5, 0.5, (batch_size, self.noise_dim))		

	def train(self, epochs, batch_size, g_iter, d_iter, model_dir, model_name):
		dp = data_processor.DataProcessor()
		sample_noise = np.random.normal(0, 1, (5*5, self.noise_dim))

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			for epoch in range(epochs):
				d_loss, g_loss = 0,0
				batched_img = dp.get_batch(batch_size, d_iter)
				
				for b, img in enumerate(batched_img):
					z = self.get_noise(batch_size)
					for it in range(d_iter):
						_, d_loss, d_real, d_fake = sess.run([ self.d_optim, self.d_loss, self.d_real_loss, self.d_fake_loss ], 
													  feed_dict={ self.g_inputs:z, self.d_inputs:img })
	# 					print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | W_dist: {:.6f}'
	# 						  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, d_iter, w_distance), end='\r')

					z = self.get_noise(batch_size)
					for it in range(g_iter):
						_, g_loss = sess.run([ self.g_optim, self.g_loss ], feed_dict={ self.g_inputs:z })

	# 					print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | G_loss: {:.6f}'
	# 						  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, g_iter, g_loss), end='\r')

					print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | D_loss: {:.6f} | G_loss: {:.6f} | d_real: {:.6f} | d_fake: {:.6f}'
						  .format(epoch+1, epochs, b+1, len(batched_img), d_loss, g_loss, d_real, d_fake) )

				if (epoch+1) % self.display_step == 0:
					samples = sess.run(self.g_sample, feed_dict={ self.g_inputs:sample_noise })
					samples = samples / 2 + 0.5
# 					print(samples[0])
					fig = self.visualize_result(samples)
					plt.savefig(self.pic_path+'{}.png'.format(str(epoch+1).zfill(4)), bbox_inches='tight')
					plt.close(fig)
					
			if model_name != '':
				self.saver.save(sess, './model/{}/{}'.format(model_name, model_name))
				print('Model saved at \'{}\''.format('./model/'+model_name))
				
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


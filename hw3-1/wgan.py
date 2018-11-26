import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import data_processor

class WGAN():
	def __init__(self, noise_dim):
		self.noise_dim = noise_dim
		self.clip_w = 0.01
		self.display_step = 10
		# plt.switch_backend('agg')
		self.build_model()

	def discriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			# inputs: (batch_size, img_h, img_w, channel)
			conv1 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(4,4)), alpha=0.01)
			conv2 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv1 , filters=64, kernel_size=(4,4)), alpha=0.01)
			conv3 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=(4,4), padding="same"), alpha=0.01)
			conv4 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(4,4)), alpha=0.01)
			flatten = tf.layers.flatten(conv4)
			# output has no activation function
			output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs, reuse=False):
		# inputs: (batch_size, noise_dim)
		with tf.variable_scope('generator', reuse=reuse):
			dense1 = tf.nn.leaky_relu(tf.layers.dense(inputs, 128*16*16))
			reshape = tf.reshape(dense1, shape=[-1, 16, 16, 128])
			conv_trans1 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(inputs=reshape, filters=128, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=0.01)
			conv_trans2 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(inputs=conv_trans1, filters=64, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=0.01)
			conv3 = tf.nn.leaky_relu(tf.layers.conv2d(inputs=conv_trans2, filters=3, kernel_size=(4,4), strides=(1,1), padding='same'), alpha=0.01)
			output = tf.nn.tanh(conv3)
		return output

	def build_model(self):
		self.g_inputs = tf.placeholder(shape=(None,self.noise_dim), dtype=tf.float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32, name="discriminator_inputs")

		self.d_loss = tf.reduce_mean( self.discriminator(self.d_inputs, tf.AUTO_REUSE) )
		self.g_loss = -tf.reduce_mean( self.discriminator(self.generator(self.g_inputs, False), tf.AUTO_REUSE) )
		self.w_distance =  self.d_loss + self.g_loss

		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

		self.d_optim = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize( -1*self.w_distance, var_list=d_vars )
		self.g_optim = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize( self.g_loss, var_list=g_vars )

		self.clip_D = [ var.assign(tf.clip_by_value(var, -self.clip_w, self.clip_w)) for var in d_vars ]
		self.fake_img = self.generator(self.g_inputs, reuse=tf.AUTO_REUSE)

	def train(self, epochs, batch_size, g_iter, d_iter, train_size=33431):
		dp = data_processor.DataProcessor(train_size)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				w_distance, g_loss = 0,0
				z = np.random.normal(0, 1, (batch_size, self.noise_dim))
				batched_img = dp.get_batch(batch_size, d_iter)
				
				for i in range(d_iter):
					_, _, w_distance = sess.run([ self.d_optim, self.clip_D, self.w_distance ], 
												  feed_dict={ self.g_inputs:z, self.d_inputs:batched_img[i] })

				z = np.random.normal(0, 1, (batch_size, self.noise_dim))
				for _ in range(g_iter):
					_, g_loss = sess.run([ self.g_optim, self.g_loss ], feed_dict={ self.g_inputs:z })

				print('Epoch [{:>2}/{:>2}] | W_dist: {:.6f} | G_loss: {:.6f}'
					  .format(epoch+1, epochs, w_distance, g_loss) )

				if (epoch+1) % self.display_step == 0:
					samples = [ sess.run(self.fake_img, feed_dict={ self.g_inputs:np.random.normal(0, 1, (5*5, self.noise_dim)) }) ]
					fig = self.visualize_result(samples)
					plt.savefig('./pics/{}.png'.format(epoch+1), bbox_inches='tight')
# 					plt.close(fig)
		
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

# 			plt.imshow(sample.reshape(64, 64, 3), cmap='Greys_r')

		return fig

		

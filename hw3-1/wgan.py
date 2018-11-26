import numpy as np
import tensorflow as tf
import data_processor

class wgan():
	def __init__(self, noise_dim):
		self.noise_dim = noise_dim
		self.epochs = epochs
		self.clip_w = 0.01

	def discriminator(self, inputs, reuse=False):
		with tf.variable_scope('discriminator', reuse=reuse):
			# inputs: (batch_size, img_h, img_w, channel)
			conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
			conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
			conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=(4,4), padding="same", activation=tf.nn.leaky_relu(alpha=0.01))
			conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
			flatten = tf.layers.flatten(conv4)
			# output has no activation function
			output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs, reuse=False):
		# inputs: (batch_size, noise_dim)
		with tf.variable_scope('generator', reuse=reuse):
			dense1 = tf.layers.dense(inputs=inputs, 128*16*16, activation=tf.nn.leaky_relu)
			reshape = tf.layers.reshape(dense1, shape=[16, 16, 128])
			conv_trans1 = tf.layers.conv2d_transpose(inputs=reshape, filters=128, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.leaky_relu(alpha=0.01))
			conv_trans2 = tf.layers.conv2d_transpose(inputs=conv_trans1, filters=64, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.leaky_relu(alpha=0.01))
			conv_trans3 = tf.layers.conv2d_transpose(inputs=conv_trans2, filters=3, kernel_size=(4,4), strides=(2,2), padding='same')
			output = tf.nn.tanh(conv_trans3)
		return output

	def build_model(self):
		self.g_inputs = tf.placeholder(shape=(None,self.noise_dim), dtype=float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=float32, name="discriminator_inputs")

		self.d_loss = tf.reduce_mean( self.discriminator(self.d_inputs, tf.AUTO_REUSE) )
		self.g_loss = -tf.reduce_mean( self.discriminator(self.generator(self.g_inputs, False), tf.AUTO_REUSE) )
		self.w_distance =  self.d_loss + self.g_loss

		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

		self.d_optim = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize( self.w_distance, var_list=d_vars )
		self.g_optim = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize( self.g_loss, var_list=g_vars )

		self.clip_D = [ var.assign(tf.clip_by_value(var, -self.clip_w, self.clip_w)) for var in d_vars ]

	def train(self, epochs, batch_size, g_iter, d_iter, train_size=33431):
		dp = data_processor.DataProcessor(train_size)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):
				batched_img = dp.get_batch(batch_size)

				for _ in range(d_iter):
					_, _, w_distance = sess.run([ self.d_optim, self.clip_D, self.w_distance ], 
												  feed_dict={ self.g_inputs= ,self.d_inputs= })

				for _ in range(g_iter):

		

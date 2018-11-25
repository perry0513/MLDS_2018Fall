import numpy as np
import tensorflow as tf

class wgan():
	def __init__(self, noise_dim, batch_size):
		self.noise_dim = noise_dim
		self.batch_size = batch_size

	def discriminator(self):
		# inputs: (batch_size, img_h, img_w, channel)
		self.discriminator_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=float32, name="discriminator_inputs")
		conv1 = tf.layers.conv2d(inputs=discriminator_inputs, filters=32, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
		conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
		conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=(4,4), padding="same", activation=tf.nn.leaky_relu(alpha=0.01))
		conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(4,4), activation=tf.nn.leaky_relu(alpha=0.01))
		flatten = tf.layers.flatten(conv4)
		# output has no activation function
		output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs):
		# inputs: (batch_size, noise_dim)
		self.generator_inputs = tf.placeholder(shape=(None,self.noise_dim), dtype=float32, name='generator_inputs')
		dense1 = tf.layers.dense(inputs=self.generator_inputs, 128*16*16, activation=tf.nn.leaky_relu)
		reshape = tf.layers.reshape(dense1, shape=[16,16,128])
		conv_trans1 = tf.layers.conv2d_transpose(inputs=reshape, filters=128, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.leaky_relu(alpha=0.01))
		conv_trans2 = tf.layers.conv2d_transpose(inputs=conv_trans1, filters=64, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.leaky_relu(alpha=0.01))
		conv_trans3 = tf.layers.conv2d_transpose(inputs=conv_trans2, filters=3, kernel_size=(4,4), strides=(2,2), padding='same')
		output = tf.nn.tanh(conv_trans3)
		return output

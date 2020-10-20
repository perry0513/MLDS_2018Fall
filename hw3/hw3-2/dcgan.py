import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import data_processor
from tqdm import tqdm
import os

class DCGAN():
	def __init__(self, noise_dim):
		self.pic_path = './dcgan_pics/'
		if not os.path.exists(self.pic_path):
			os.makedirs(self.pic_path)
		self.noise_dim = noise_dim
		self.alpha = 0.2
		self.learning_rate = 0.0005
		self.beta1 = 0.5
		self.dropout_rate = 0.2
		self.display_step = 1
		
		self.test_labels = [np.zeros(23) for _ in range(25)]
		for i in range( 0, 5):
			self.test_labels[i][9] = 1
			self.test_labels[i][15] = 1
		for i in range( 5,10):
			self.test_labels[i][9] = 1
			self.test_labels[i][21] = 1
		for i in range(10,15):
			self.test_labels[i][9] = 1
			self.test_labels[i][7] = 1
		for i in range(15,20):
			self.test_labels[i][4] = 1
			self.test_labels[i][15] = 1
		for i in range(20,25):
			self.test_labels[i][4] = 1
			self.test_labels[i][7] = 1

		plt.switch_backend('agg')
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=50)

	def discriminator(self, inputs, condition, reuse=False, is_training=True):
		with tf.variable_scope('discriminator', reuse=reuse):
			# inputs: (batch_size, img_h, img_w, channel)
			conv1 = tf.nn.leaky_relu((tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(5,5), strides=(2,2), padding="SAME",
													   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))), alpha=self.alpha)
# 			conv1 = tf.layers.dropout(conv1, rate=self.dropout_rate, training=is_training)
			conv2 = tf.nn.leaky_relu((tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=(5,5), strides=(2,2), padding="SAME",
													   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))), alpha=self.alpha)
# 			conv2 = tf.layers.dropout(conv2, rate=self.dropout_rate, training=is_training)
				
			conv3 = tf.nn.leaky_relu((tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=(5,5), strides=(2,2), padding="SAME",
													   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))), alpha=self.alpha)
			embed_y = tf.expand_dims(condition, 1)
			embed_y = tf.expand_dims(embed_y, 2)
			tiled_embeddings = tf.tile(embed_y, [1, 8, 8, 1])
			concat = tf.concat([conv3, tiled_embeddings], -1)
# 			conv3 = tf.layers.dropout(conv3, rate=self.dropout_rate, training=is_training)
			conv4 = tf.nn.leaky_relu((tf.layers.conv2d(inputs=concat, filters=256, kernel_size=(1,1), strides=(1,1), padding="SAME",
													   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))), alpha=self.alpha)
# 			conv4 = tf.layers.dropout(conv4, rate=self.dropout_rate, training=is_training)
			flatten = tf.layers.flatten(conv4)
			# output has no activation function
			output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs, condition, reuse=False, is_training=True):
		# inputs: (batch_size, noise_dim)
		with tf.variable_scope('generator', reuse=reuse):
			inputs = tf.concat([inputs, condition], -1)
			dense1 = tf.nn.leaky_relu(tf.layers.dense(inputs, 128*16*16))
			reshape = tf.reshape(dense1, shape=[-1, 16, 16, 128])
			conv_trans1 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(reshape, filters=128, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
# 			conv_trans1 = tf.layers.dropout(conv_trans1, rate=self.dropout_rate, training=is_training)
			conv_trans2 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(conv_trans1, filters=64, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
# 			conv_trans2 = tf.layers.dropout(conv_trans2, rate=self.dropout_rate, training=is_training)
			conv3 = tf.nn.leaky_relu(tf.layers.conv2d(conv_trans2, filters=3, kernel_size=(4,4), strides=(1,1), padding='same'), alpha=self.alpha)
# 			conv3 = tf.layers.dropout(conv3, rate=self.dropout_rate, training=is_training)
			output = tf.nn.tanh(conv3)
		return output

	def build_model(self):
		print('Building model . . .')
		self.g_inputs = tf.placeholder(shape=(None, self.noise_dim), dtype=tf.float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32, name="discriminator_inputs")
		self.label = tf.placeholder(shape=(None, 23), dtype=tf.float32, name='train_label')
		self.wrong_label = tf.placeholder(shape=(None, 23), dtype=tf.float32, name='train_wrong_label')

		self.g_sample = self.generator(self.g_inputs, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_real = self.discriminator(self.d_inputs, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_fake = self.discriminator(self.g_sample, self.label, tf.AUTO_REUSE, is_training=True)
# 		self.d_wrong_img = self.discriminator(self.wrong_img, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_wrong_label = self.discriminator(self.d_inputs, self.wrong_label, tf.AUTO_REUSE, is_training=True)
		
		# add noise to labels 
# 		true_label = tf.random_uniform(tf.shape(self.d_fake), 0.8, 1.2)
# 		false_label = tf.random_uniform(tf.shape(self.d_fake), 0, 0.3)
		self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real)))
		self.d_fake_loss = ( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake))) + \
							 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_wrong_label, labels=tf.zeros_like(self.d_wrong_label))) ) / 2
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))
		self.d_loss = self.d_real_loss + self.d_fake_loss
		
		self.g_infer = self.generator(self.g_inputs, self.label, tf.AUTO_REUSE, is_training=False)
		
		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

		self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.d_loss, var_list=d_vars )
		self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.g_loss, var_list=g_vars )


	def get_noise(self, batch_size):
# 		return np.random.normal(0, 1, (batch_size, self.noise_dim))
		noise = np.random.uniform(-0.5, 0.5, (batch_size, self.noise_dim))
		norm = np.linalg.norm(noise, axis=1)
		noise = [ noise[i] / norm[i] for i in range(batch_size) ]
		return noise
		
	def train(self, epochs, batch_size, g_iter, d_iter, model_dir, model_name):
		print('Training . . .')
		dp = data_processor.DataProcessor()
		sample_noise = self.get_noise(5*5)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			for epoch in range(epochs):
				d_loss, g_loss = 0,0
				batched_img, batched_label, batched_wrong_label = dp.get_batch(batch_size, d_iter)
				dataset = list(zip(batched_img, batched_label, batched_wrong_label))
				
				for b, (img, label, wrong_label) in enumerate(dataset):
					z = self.get_noise(batch_size)
					for it in range(d_iter):
						_, d_loss, d_real, d_fake = sess.run([ self.d_optim, self.d_loss, self.d_real_loss, self.d_fake_loss ], 
													  feed_dict={ self.g_inputs:z, self.d_inputs:img, self.label:label, self.wrong_label:wrong_label })
# 						print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | W_dist: {:.6f}'
# 							  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, d_iter, w_distance), end='\r')
					
					z = self.get_noise(batch_size)
					for it in range(g_iter):
						_, g_loss = sess.run([ self.g_optim, self.g_loss ], feed_dict={ self.g_inputs:z, self.label:label })
						
# 						print('Epoch [{:>2}/{:>2}] | Batch [{:>3}/{:>3}] | Iter [{:>2}/{:>2}] | G_loss: {:.6f}'
# 							  .format(epoch+1, end_epoch, b+1, len(batched_img), it+1, g_iter, g_loss), end='\r')

					print('Epoch [{:>2}/{:>2}] | Batch [{:>4}/{:>4}] | D_loss: {:.6f} | G_loss: {:.6f} | d_real: {:.6f} | d_fake: {:.6f}'
						  .format(epoch+1, epochs, b+1, len(batched_img), d_loss, g_loss, d_real, d_fake) )

				if (epoch+1) % self.display_step == 0:
					samples = sess.run(self.g_infer, feed_dict={ self.g_inputs:sample_noise, self.label:self.test_labels })
					samples = samples / 2 + 0.5
					fig = self.visualize_result(samples)
					plt.savefig(self.pic_path+'{}.png'.format(str(epoch+1).zfill(4)), bbox_inches='tight')
					plt.close(fig)
			
			if model_name != '':
# 				if os.path.exists('./model')
				self.saver.save(sess, './model/{}/{}'.format(model_name, model_name))
				print('Model saved at \'{}\''.format('./model/'+model_name))
			
	def infer(self, model_dir):
		sample_noise = self.get_noise(5*5)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			samples = sess.run(self.g_infer, feed_dict={ self.g_inputs:sample_noise, self.label:self.test_labels })
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


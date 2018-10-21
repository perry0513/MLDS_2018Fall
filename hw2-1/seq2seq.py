import numpy as np
import tensorflow as tf
import helpers
import json
from pprint import pprint
import skvideo.io 

tf.reset_default_graph()
sess = tf.InteractiveSession()

class Seq2seq():
	def __init__(self, rnn_size, num_layers, feat_size, batch_size, word_to_idx, 
				 mode, max_encoder_steps, max_decoder_steps, embedding_size):
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.feat_size = feat_size
		self.
		self.word_to_idx = word_to_idx
		self.mode = mode
		self.max_encoder_steps = max_encoder_steps
		self.max_decoder_steps = max_decoder_steps
		self.embedding_size = embedding_size

		self.vocab_size = len(self.word_to_idx)

        self.build_model()

	def build_model(self):
		# Model input & output

		# shape=(batch_size, encoder_hidden_units, input_size)
		self.encoder_inputs  = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='encoder_inputs')
		# shape=(batch_size, num_of_sentence, decoder_hidden_units)
		self.decoder_targets = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='decoder_targets') 
		self.decoder_inputs  = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='decoder_inputs')
		# embeddings

		embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
		np.transpose(decoder_inputs_embedded, [1,0,2])

		# Encoder

		encoder_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)

		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
			encoder_cell,
			encoder_inputs,
			dtype = tf.float32,
			time_major = True
			)

		# Decoder

		decoder_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, use_peepholes=True)


		output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

		self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
		max_target_sequence_length = tf.reduce_max(decoder_targets_length, name='max_target_len')
		mask = tf.sequence_mask(decoder_targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')



		# Helper
		training_helper = tf.contrib.seq2seq.TrainingHelper(
										inputs=decoder_inputs_embedded, 
										sequence_length=decoder_targets_length, 
										time_major=False, name='training_helper')

		# Decoder
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
										cell=decoder_cell, helper=training_helper, 
										initial_state=decoder_initial_state, 
										output_layer=output_layer)

		decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
										decoder=training_decoder, 
										impute_finished=True, 
										maximum_iterations=self.max_target_sequence_length)


		decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
		decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')

		# Optimizer

		loss = tf.contrib.seq2seq.sequence_loss(
							logits=decoder_logits_train, 
							targets=decoder_targets, 
							weights=mask)
		optimizer = tf.train.AdamOptimizer()

		train_op = optimizer.minimize(loss)

	def train(self, sess, encoder_inputs):






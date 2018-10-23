import numpy as np
import tensorflow as tf
import store_data
import json
from pprint import pprint
import skvideo.io 

tf.reset_default_graph()
sess = tf.InteractiveSession()

class Seq2seq():
	def __init__(self, rnn_size, num_layers, feat_size, batch_size, vocab_size, 
				 mode, max_encoder_steps, max_decoder_steps, embedding_size):
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.feat_size = feat_size
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.mode = mode
		self.max_encoder_steps = max_encoder_steps
		self.max_decoder_steps = max_decoder_steps
		self.embedding_size = embedding_size

		self.build_model()

	def build_model(self):
		# Model input & output

		# shape=(batch_size, encoder_hidden_units, input_size)
		self.encoder_inputs  = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='encoder_inputs')
		# shape=(batch_size, num_of_sentence, decoder_hidden_units)
		self.decoder_targets = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='decoder_targets') 
		self.decoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
		# embeddings

		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
		tf.transpose(decoder_inputs_embedded, [1,0,2])

		# Encoder

		encoder_cell = self._create_cell()

		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype = tf.float32, time_major = True)

		# Decoder

		# Decoder cell with attention
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
													num_units=self.rnn_size, 
													memory=encoder_outputs, 
													normalize=True)

		decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
												cell=self._create_cell(), 
												attention_mechanism=attention_mechanism, 
												attention_layer_size=self.rnn_size, 
												name='Attention_Wrapper')

		decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_final_state)

		output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))


		self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
		max_target_sequence_length = tf.reduce_max(decoder_targets_length, name='max_target_len')
		mask = tf.sequence_mask(decoder_targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')



		# Define helper based on 'train' or 'infer' mode
		if self.mode == 'train':
			training_helper = tf.contrib.seq2seq.TrainingHelper(
											inputs=decoder_inputs_embedded, 
											sequence_length=decoder_targets_length,
											# memory_sequence_length=?????????, 
											time_major=True, name='training_helper')
		elif self.mode == 'infer':
			training_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
											embedding=embeddings, 
											start_tokens=tf.fill([self.batch_size], 1), # word_to_idx['<BOS>'] = 1
											end_token=2) 								# word_to_idx['<EOS>'] = 2

		# Decoder
		training_decoder = tf.contrib.seq2seq.BasicDecoder(
										cell=decoder_cell, helper=training_helper, 
										initial_state=decoder_initial_state, 
										output_layer=output_layer)

		decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
										decoder=training_decoder, 
										impute_finished=True, 
										maximum_iterations=self.max_target_sequence_length)

		# Calculate loss with sequence_loss
		decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
		self.decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')


		self.loss = tf.contrib.seq2seq.sequence_loss(
							logits=decoder_logits_train, 
							targets=decoder_targets, 
							weights=mask)
		optimizer = tf.train.AdamOptimizer()

		# Clip gradient if gradient is too large
		gradients = optimizer.compute_gradients(loss)
		capped_gradients = [ (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None ]
		self.train_op = optimizer.apply_gradients(capped_gradients)

		tf.summary.scalar('loss', self.loss)
		self.summary_op = tf.summary.merge_all()


	def _create_cell(self):
		def single_cell():
			cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
			# cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder, seed=9487)
			return cell
		cell = tf.contrib.rnn.MultiRNNCell([ single_cell() for _ in range(self.num_layers) ])
		return cell


	def train(self, sess, encoder_inputs, decoder_inputs, decoder_targets, decoder_targets_length):
		feed_dict = { self.encoder_inputs : encoder_inputs,
					  self.decoder_inputs : decoder_inputs,
					  self.decoder_targets : decoder_targets,
					  self.decoder_targets_length : decoder_targets_length }

		_, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
		return loss, summary

'''
	def test_predict(self, sess, encoder_inputs, decoder_inputs, decoder_targets, decoder_targets_length):
		feed_dict = { self.encoder_inputs = encoder_inputs,
					  self.decoder_inputs = decoder_inputs,
					  self.decoder_targets = decoder_targets,
					  self.decoder_targets_length = decoder_targets_length }

		predict = sess.run(decoder_logits_train, feed_dict)
		return predict
'''
 


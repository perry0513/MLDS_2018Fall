import numpy as np
import tensorflow as tf
# from pprint import pprint
# import skvideo.io 

tf.reset_default_graph()
sess = tf.InteractiveSession()

class Seq2seq():
	def __init__(self, rnn_size, num_layers, feat_size, batch_size, vocab_size, 
				 mode, max_encoder_steps, max_decoder_steps, embedding_size, beam_size=0):
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.feat_size = feat_size
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.mode = mode
		self.max_encoder_steps = max_encoder_steps
		self.max_decoder_steps = max_decoder_steps
		self.beam_size = beam_size
		self.embedding_size = embedding_size

		self.build_model()

		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

	def build_model(self):
		# Model input & output

		# shape = (batch_size, encoder_hidden_units, input_size)
		self.encoder_inputs  = tf.placeholder(shape=(None, None, 4096), dtype=tf.float32, name='encoder_inputs')
		# shape = (num_of_sentence, decoder_hidden_units)
		self.decoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets') 
		self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

		# embeddings

		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
		decoder_inputs_embedded = tf.transpose(decoder_inputs_embedded, [1,0,2])

		# Encoder

		encoder_cell = self._create_cell()

		encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype = tf.float32, time_major = True)
		encoder_outputs = tf.transpose(encoder_outputs, [1,0,2]) # [batch_size, 80, 1024]
		if self.mode == 'test':
			encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
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

			

		output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))


		max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
		mask = tf.sequence_mask(self.decoder_targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')



		# Define helper based on 'train' or 'infer' mode
		if self.mode == 'train':
			decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_final_state)

			training_helper = tf.contrib.seq2seq.TrainingHelper(
											inputs=decoder_inputs_embedded, 
											sequence_length=self.decoder_targets_length,
											# memory_sequence_length=?????????, 
											time_major=True, name='training_helper')
			# Decoder
			training_decoder = tf.contrib.seq2seq.BasicDecoder(
											cell=decoder_cell, helper=training_helper, 
											initial_state=decoder_initial_state, 
											output_layer=output_layer)

			decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
											decoder=training_decoder, 
											output_time_major=False,
											impute_finished=True, 
											maximum_iterations=max_target_sequence_length)

			# Calculate loss with sequence_loss
			print ('decoder_outputs.rnn_output: ', decoder_outputs.rnn_output)
			print ('decoder_outputs.sample_id: ', decoder_outputs.sample_id)
			decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
			self.decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')
			# decoder_logits_train = tf.transpose(decoder_logits_train, [1,0,2])
			

			print ('decoder_logits_train: ',decoder_logits_train.shape)
			print ('decoder_targets: ',self.decoder_targets.shape)
			self.loss = tf.contrib.seq2seq.sequence_loss(
								logits=decoder_logits_train, 
								targets=self.decoder_targets, 
								weights=mask)
			optimizer = tf.train.AdamOptimizer()

			# Clip gradient if gradient is too large
			gradients = optimizer.compute_gradients(self.loss)
			capped_gradients = [ (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None ]
			self.train_op = optimizer.apply_gradients(capped_gradients)


		else:
			# inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			# 								embedding=embeddings, 
			# 								start_tokens=tf.fill([self.batch_size], 1), 
			# 								end_token=2) 								# <EOS> = 2

			# 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
			# encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)

			# encoder_final_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_final_state)
			self.tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=self.beam_size)

			batch_size = self.batch_size * self.beam_size

			decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=self.tiled_encoder_final_state)


			inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
											cell=decoder_cell,
											embedding=embeddings,
											start_tokens=tf.fill([self.batch_size, ], 1),	# <BOS> = 1
											end_token=2, 									# <EOS> = 2
											initial_state=decoder_initial_state,
											beam_width=self.beam_size,
											output_layer=output_layer)


			inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
															decoder=inference_decoder,
															maximum_iterations=self.max_decoder_steps)

			self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
			self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output



			


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

		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		return loss


	def infer(self, sess, encoder_inputs):
		feed_dict = { self.encoder_inputs : encoder_inputs }

		# a = sess.run(self.tiled_encoder_final_state, feed_dict=feed_dict)
		# print(a[0][1].shape)

		predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
		print(predict.shape)
		return predict, logits



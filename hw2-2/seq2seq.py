import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec


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
		self.encoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.float32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_inputs_length')
		# shape = (num_of_sentence, decoder_hidden_units)
		self.decoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets') 
		self.decoder_targets_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_targets_length')

		self.keep_prob_placeholder = tf.placeholder(dtype=tf.float32)
		self.sampling_probability = tf.placeholder(dtype=tf.float32)



		# embeddings (gensin word2vec)
		w2v_model = Word2Vec.load("word2vec.model")
		w2v_embedding = np.zeros([len(w2v_model.wv.index2word), self.embedding_size])
		for idx, word in enumerate(w2v_model.wv.index2word):
			w2v_embedding[idx, :] = w2v_model[word]
		embeddings = tf.Variable(tf.convert_to_tensor(w2v_embedding, np.float32), trainable=False, name='embedding')

		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
#####??????
		encoder_inputs_embedded = tf.transpose(encoder_inputs_embedded, [1,0,2])
#####??????
		# embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), dtype = tf.float32)


		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
		decoder_inputs_embedded = tf.transpose(decoder_inputs_embedded, [1,0,2])

		# Encoder

		# encoder_cell = self._create_cell()
		# encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype = tf.float32, time_major = True)
#####???
		encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
														cell_fw=self._create_cell(),
														cell_bw=self._create_cell(), 
														inputs=self.encoder_inputs,
														sequence_length=self.encoder_inputs_length,
														dtype=tf.float32,
														time_major=True )
#####???

		encoder_outputs = tf.transpose(encoder_outputs, [1,0,2]) # shape(batch_size,max_step,rnn_size)
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
#####???
			training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
											inputs=decoder_inputs_embedded,
											sequence_length=self.decoder_targets_length,
											embedding=embeddings,
											sampling_probability=self.sampling_probability,
											time_major=True)
#####???
			# training_helper = tf.contrib.seq2seq.TrainingHelper(
			# 								inputs=decoder_inputs_embedded, 
			# 								sequence_length=self.decoder_targets_length,
			# 								# memory_sequence_length=?????????, 
			# 								time_major=True, name='training_helper')
			# Decoder
			training_decoder = tf.contrib.seq2seq.BasicDecoder(
											cell=decoder_cell, helper=training_helper, 
											initial_state=decoder_initial_state, 
											output_layer=output_layer)

			decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
											decoder=training_decoder, 
											impute_finished=True, 
											maximum_iterations=max_target_sequence_length)

			# Calculate loss with sequence_loss
			decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
			self.decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')
			# decoder_logits_train = tf.transpose(decoder_logits_train, [1,0,2])
			

			self.loss = tf.contrib.seq2seq.sequence_loss(
									logits=decoder_logits_train, 
									targets=self.decoder_targets, 
									weights=mask)
			optimizer = tf.train.AdamOptimizer()

			# Clip gradient if gradient is too large
			gradients = optimizer.compute_gradients(self.loss)
			capped_gradients = [ (tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in gradients if grad is not None ]
			self.train_op = optimizer.apply_gradients(capped_gradients)


		else:
			# 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
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
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob_placeholder)
			return cell
		cell = tf.contrib.rnn.MultiRNNCell([ single_cell() for _ in range(self.num_layers) ])
		return cell


	def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, 
			  decoder_targets, decoder_targets_length, sampling_probability):
		feed_dict = { self.encoder_inputs : encoder_inputs,
					  self.encoder_inputs_length : encoder_inputs_length,
					  self.decoder_inputs : decoder_inputs,
					  self.decoder_targets : decoder_targets,
					  self.decoder_targets_length : decoder_targets_length,
					  self.sampling_probability : sampling_probability,
					  self.keep_prob_placeholder : 0.8 }

		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
		return loss


	def infer(self, sess, encoder_inputs):
		feed_dict = { self.encoder_inputs : encoder_inputs,
					  self.keep_prob_placeholder : 1.0 }

		predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
		return predict, logits



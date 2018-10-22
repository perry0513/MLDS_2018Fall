

import numpy as np
import tensorflow as tf
import store_data
import tadpol_helper
from seq2seq import Seq2seq
import time


encoder_input_video = store_data.get_video()
idx_to_word, indexed_sentence = store_data.get_sentence()

epochs = 5
batch_size = 5

rnn_size   = 1024
num_layers = 1
feat_size  = 4096
vocab_size = len(idx_to_word)
max_encoder_steps = 80
max_decoder_steps = 50
embedding_size = rnn_size





with tf.Session() as sess:
	model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, feat_size=feat_size, batch_size=batch_size, vocab_size=vocab_size, 
					mode='infer', max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, embedding_size=embedding_size)

	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print ('Reloading model parameters...')
		model.saver.restore(sess, ckpt.model_checkpoint_path)

	# for epoch in epochs:
	# 	shuffled_video, shuffled_sentences = tadpol_helper.shuffle_and_zip(indexed_sentence, encoder_input_video, batch_size)
	# 	trainset = list(zip( shuffled_video, shuffled_sentences ))

	# 	for step, (batch_video, batch_sentences) in enumerate(trainset):
	# 		np.transpose(batch_video, [1,0,2])
	# 		model.train(sess, batch_video, batch_sentences)


	



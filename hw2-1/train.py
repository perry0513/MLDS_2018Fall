import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
import time


# Hyper-parameters
epochs = 5
batch_size = 5

mode = 'test'
rnn_size   = 1024
num_layers = 1
feat_size  = 4096
max_encoder_steps = 80
max_decoder_steps = 50
embedding_size = 128

data_processor = DataProcessor(mode)
idx2word_dict = data_processor.get_dictionary()
vocab_size = len(idx2word_dict)


model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, feat_size=feat_size, batch_size=batch_size, vocab_size=vocab_size, 
				mode=mode, max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, embedding_size=embedding_size)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length = data_processor.get_shuffle_and_batch(batch_size)

		trainset = list(zip( encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length ))

		for step, (batch_videos, batch_dec_inputs, batch_dec_targets, batch_dec_targets_len) in enumerate(trainset):
			batch_videos = np.transpose(batch_videos, [1,0,2])
			loss, summary = model.train(sess=sess, encoder_inputs=batch_videos, decoder_inputs=batch_dec_inputs,
										decoder_targets=batch_dec_targets , decoder_targets_length=batch_dec_targets_len )
			print ('loss: ', loss)


	model.saver.save(sess, './model/' + time.strftime("%m%d%Y_%H%M", time.localtime()))



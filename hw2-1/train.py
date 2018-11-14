import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
from tqdm import tqdm


# Hyper-parameters
<<<<<<< HEAD
epochs = 5
batch_size = 10 
=======
epochs = 40
batch_size = 25
>>>>>>> cafaec0cc52506058334166867561fa32f88d4ec

mode = 'train'
rnn_size   = 1024
num_layers = 1
feat_size  = 4096
max_encoder_steps = 80
max_decoder_steps = 50
embedding_size = 256

data_processor = DataProcessor(mode)
idx2word_dict = data_processor.get_dictionary()
vocab_size = len(idx2word_dict)


model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, feat_size=feat_size, batch_size=batch_size, vocab_size=vocab_size, 
				mode=mode, max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, embedding_size=embedding_size)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length = data_processor.get_batch(batch_size, shuffle=True)

		trainset = list(zip( encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length ))

		for step, (batch_videos, batch_dec_inputs, batch_dec_targets, batch_dec_targets_len) in enumerate(trainset):
			batch_videos = np.transpose(batch_videos, [1,0,2])
			loss = model.train(sess=sess, encoder_inputs=batch_videos, decoder_inputs=batch_dec_inputs,
							   decoder_targets=batch_dec_targets , decoder_targets_length=batch_dec_targets_len)
			print ('Epoch: {:>2} | Step: {:>3} | Loss: {:.6f}'.format(epoch+1, step+1, loss))

##### Validate #####
	# encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length = data_processor.get_batch(batch_size, shuffle=False)
	# validset = list(zip( encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length ))
	# vid_num = 0
	# for batch_videos, batch_dec_inputs, batch_dec_targets, batch_dec_targets_len in validset:
	# 	batch_videos = np.transpose(batch_videos, [1,0,2])
	# 	predict = model.validate(sess=sess, encoder_inputs=batch_videos, decoder_inputs=batch_dec_inputs,
	# 							 decoder_targets=batch_dec_targets , decoder_targets_length=batch_dec_targets_len)


##########
	model.saver.save(sess, './model/'+'ep_{}_bs_{}_rnn_{}_lay_{}_emb_{}'.format(epochs, batch_size, rnn_size, num_layers, embedding_size))



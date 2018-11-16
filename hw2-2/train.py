import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
from tqdm import tqdm


# Hyper-parameters
epochs = 10
batch_size = 50

mode = 'train'
rnn_size   = 1024
num_layers = 1
max_encoder_steps = 30
max_decoder_steps = 30
embedding_size = 256

data_processor = DataProcessor(mode)
idx2word_dict = data_processor.get_dictionary()
vocab_size = len(idx2word_dict)


model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, batch_size=batch_size, vocab_size=vocab_size, mode=mode, 
				max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, embedding_size=embedding_size)

# TODO: sampling probability for each epoch
def func(x):
	return 2 - 2 / (1 + np.exp(0.3*(x-2*epochs)))
sampling_prob = func(np.arange(epochs))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_targets, decoder_targets_length = data_processor.get_batch(batch_size)
		trainset = list(zip( encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_targets, decoder_targets_length ))

		for step, (batch_enc_inputs, batch_enc_inputs_length, batch_dec_inputs, batch_dec_targets, batch_dec_targets_len) in enumerate(trainset):
			# trim batch_dec_targets to max time step
			max_target_len = max(batch_dec_targets_len)
			batch_dec_targets = [ line[:max_target_len] for line in batch_dec_targets ]

			loss = model.train(sess=sess, encoder_inputs=batch_enc_inputs, encoder_inputs_length=batch_enc_inputs_length,
							   decoder_inputs=batch_dec_inputs, decoder_targets=batch_dec_targets, 
							   decoder_targets_length=batch_dec_targets_len, sampling_probability=sampling_prob[epoch])
			print ('Epoch: {:>2} | Step: {:>3} | Loss: {:.6f}'.format(epoch+1, step+1, loss))


	model.saver.save(sess, './model/'+'ep_{}_bs_{}_rnn_{}_lay_{}_emb_{}'.format(epochs, batch_size, rnn_size, num_layers, embedding_size))



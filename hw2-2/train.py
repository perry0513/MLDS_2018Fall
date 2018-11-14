import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
from tqdm import tqdm


# Hyper-parameters
epochs = 40
batch_size = 25

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

# sampling probability for each epoch
sampling_prob = [ 0 for x in range(epochs) ]


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(epochs):
		encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_targets, 
		decoder_targets_length = data_processor.get_batch(batch_size, shuffle=True)

		trainset = list(zip( encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_targets, decoder_targets_length ))

		for step, (batch_enc_inputs, batch_enc_inputs_length, batch_dec_inputs, batch_dec_targets, batch_dec_targets_len) in enumerate(trainset):
			batch_enc_inputs = np.transpose(batch_enc_inputs, [1,0,2])
			loss = model.train(sess=sess, encoder_inputs=batch_enc_inputs, encoder_inputs_length=batch_enc_inputs_length,
							   decoder_inputs=batch_dec_inputs, decoder_targets=batch_dec_targets, 
							   decoder_targets_length=batch_dec_targets_len, sampling_probability=sampling_prob[step])
			print ('Epoch: {:>2} | Step: {:>3} | Loss: {:.6f}'.format(epoch+1, step+1, loss))


	model.saver.save(sess, './model/'+'ep_{}_bs_{}_rnn_{}_lay_{}_emb_{}'.format(epochs, batch_size, rnn_size, num_layers, embedding_size))



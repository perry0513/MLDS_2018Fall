import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
import time


# Hyper-parameters
epochs = 1
batch_size = 10 

mode = 'test'
rnn_size   = 1024
num_layers = 1
feat_size  = 4096
max_encoder_steps = 80
max_decoder_steps = 50
beam_size = 3
embedding_size = 128

data_processor = DataProcessor(mode)
idx2word_dict = data_processor.get_dictionary()
vocab_size = len(idx2word_dict)

model_dir = './model/'



with tf.Session() as sess:
	model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, feat_size=feat_size, batch_size=batch_size, vocab_size=vocab_size, mode=mode, 
					max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, beam_size=beam_size, embedding_size=embedding_size)

	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print ('Reloading model parameters...')
		model.saver.restore(sess, ckpt.model_checkpoint_path)

	for epoch in range(epochs):
		encoder_videos, decoder_inputs, decoder_targets, decoder_targets_length = data_processor.get_shuffle_and_batch(batch_size)

		for step, batch_videos in enumerate(encoder_videos):
			batch_videos = np.transpose(batch_videos, [1,0,2])
			predict, logits = model.infer(sess=sess, encoder_inputs=batch_videos)
			predict = np.transpose(predict, [0,2,1])
			for batch in predict:
				for beam in batch:
					sentence = [ idx2word_dict[word] for word in beam ]
					print ( ' '.join(sentence) )





	



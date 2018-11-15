import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from seq2seq import Seq2seq
from tqdm import tqdm


# Hyper-parameters
batch_size = 50

mode = 'test'
rnn_size   = 1024
num_layers = 1
max_encoder_steps = 30
max_decoder_steps = 30
beam_size = 3
embedding_size = 256

data_processor = DataProcessor(mode)
idx2word_dict = data_processor.get_dictionary()
vocab_size = len(idx2word_dict)

model_dir = './model/'

with tf.Session() as sess:
	model = Seq2seq(rnn_size=rnn_size, num_layers=num_layers, batch_size=batch_size, vocab_size=vocab_size, mode=mode, 
					max_encoder_steps=max_encoder_steps, max_decoder_steps=max_decoder_steps, beam_size=beam_size, embedding_size=embedding_size)

	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print ('Reloading model parameters...')
		model.saver.restore(sess, ckpt.model_checkpoint_path)

	encoder_inputs = data_processor.get_batch_infer_data(batch_size)

	for step, batch_inputs in enumerate(tqdm(encoder_videos)):
		batch_inputs = np.transpose(batch_inputs, [1,0])
		predict, logits = model.infer(sess=sess, encoder_inputs=batch_inputs)
		predict = np.transpose(predict, [0,2,1])
		for batch in predict:
			sentence = [ idx2word_dict[word] for word in batch[0] ]
			sentence = ' '.join(sentence)
			sentence = sentence.strip(' <EOS>')
			
			output_file.write(sentence + '\n')





	



import numpy as np
import tensorflow as tf
import helpers
import json
from pprint import pprint
from gensim.corpora import Dictionary
import skvideo.io 

tf.reset_default_graph()
sess = tf.InteractiveSession()

#PARAMETERS
PAD = 0
EOS = 1
vocab_size = 6000
input_size = 4096
encoder_hidden_units = 80
decoder_hidden_units = encoder_hidden_units

#Model input & output

# shape=(batch_size, encoder_hidden_units, input_size)
encoder_inputs = tf.placeholder(shape=(None,None,None), dtype=tf.float32, name='encoder_inputs')
# shape=(batch_size, num_of_sentence, decoder_hidden_units)
decoder_targets = tf.placeholder(shape=(None,None,None), dtype=tf.int32, name='decoder_targets') 
decoder_inputs = tf.placeholder(shape=(None,None,None), dtype=tf.int32, name = 'decoder_inputs')

# embeddings

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype = tf.float32)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

#Encoder

encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
	encoder_cell,
	encoder_inputs,
	dtype = tf.float32,
	time_major = True
	)

#Decoder

decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
	decoder_cell,
	decoder_inputs_embedded,
	initial_state = encoder_final_state,
	dtype = tf.float32,
	time_major = True,
	scope = 'plain_decoder'
	)

output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
max_target_sequence_length = tf.reduce_max(decoder_targets_length, name='max_target_len')
mask = tf.sequence_mask(decoder_targets_length, max_target_sequence_length, dtype=tf.float32, name='masks')

# Helper
	training_helper = tf.contrib.seq2seq.TrainingHelper(
		inputs=decoder_inputs_embedded, 
		sequence_length=decoder_targets_length, 
		time_major=False, name='training_helper')
# Decoder
	training_decoder = tf.contrib.seq2seq.BasicDecoder(
		cell=decoder_cell, helper=training_helper, 
		initial_state=decoder_initial_state, 
		output_layer=output_layer)



decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')

#Optimizer

loss = tf.contrib.seq2seq.sequence_loss(
                logits=decoder_logits_train, 
                targets=decoder_targets, 
                weights=mask)
optimizer = tf.train.AdamOptimizer()



# Load data


def get_train_X():
	print("Loading training features . . .")

	train_path = './MLDS_hw2_1_data/training_data/'
	train_filename = open(train_path+'id.txt', 'r').read().split('\n')
	train_filename = [ train_path+'feat/'+file+'.npy' for file in train_filename ]
	train_filename.pop()
	#print(train_filename)

	arr = np.array([np.load(file) for file in train_filename])

#	arr = 18*arr

	return arr

def get_train_Y():
	print("Loading training lables . . .")
	train_label_dict = json.loads(open('./MLDS_hw2_1_data/training_label.json', 'r').read())
	data = [ label['caption'] for label in train_label_dict ]

	caption = []
	for i in range(np.shape(data)[0]):#every video 1450
		per_video = []
		for strr in data[i]:
			temp = strr.split()
			per_video.append(temp)
		caption.append(per_video)

	dct = Dictionary(caption[0])
	for i in range(1,1450):
		dct.add_documents(caption[i])

	caption_num = []
	for i in range(1,1450): #every video 1450
		per_video = []
		r = np.shape(caption[i])[0]
		for j in range(80):

			temp = dct.doc2idx(caption[i][j%r])

			per_video.append(temp)
		caption_num.append(per_video)
	return caption_num, dct


#Train

train_X = get_train_X()
train_Y, dct = get_train_Y()


def next_feed(_idx):
	x = train_X[_idx]
	y = train_Y[_idx]


	encoder_inputs_, _ = helpers.batch(x)
	decoder_targets_, _ = helpers.batch(y)
	
	_idx = _idx + 1

	print(encoder_inputs_)

	fdict = {
		encoder_inputs: encoder_inputs_,
		decoder_targets: decoder_targets_
	}

	return fdict




loss_track = []

max_batches = 1000
batches_in_epochs = 1
_idx = 0

try:
	for batch in range(max_batches):
		fd = next_feed(_idx)

		_, l = sess.run([train_op, loss],feed_dict = fd)

		loss_track.append(l)

		if batch == 0 or batch % batches_in_epochs == 0:
			print('batch {}'.format(batch))
			print('  minibatch loss: {}'.format(sess.run(loss, fd)))
			predict_ = sess.run(decoder_prediction, fd)
			for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
				print('  sample {}:'.format(i + 1))
				print('	input	 > {}'.format(inp))
				print('	predicted > {}'.format(pred))
				if i >= 2:
					break
			print()

except KeyboardInterrupt:
	print('training interrupted')

import matplotlib.pyplot as plt
plt.plot(loss_track)
plt.show()
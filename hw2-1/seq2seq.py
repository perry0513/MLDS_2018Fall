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
vocab_size = 200
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

#Model input & output

encoder_inputs = tf.placeholder(shape=(None,None,None), dtype=tf.float32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None,None), dtype=tf.int32, name='decoder_targets') 

#Embeddings

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype = tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

#Encoder

encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
	encoder_cell,
	encoder_inputs_embedded,
	dtype = tf.float32,
	time_major = True
	)
del encoder_outputs

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

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(decoder_logits, 2)


#Optimizer

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
	labels = tf.one_hot(decoder_targets, depth = vocab_size, dtype = tf.float32),
	logits = decoder_logits
	)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


#Prepare data
# Load data
'''
train_id    = './MLDS_hw2_1_data/training_data/id.txt'
train_path  = './MLDS_hw2_1_data/training_data/feat/'
train_label = './MLDS_hw2_1_data/training_label.json'
test_id     = './MLDS_hw2_1_data/testing_data/id.txt'
test_path   = './MLDS_hw2_1_data/testing_data/feat/'
test_label  = './MLDS_hw2_1_data/testing_label.json'

def load_data():
	train_filename = open(train_id, 'r').read().splitlines()
	train_filename = [ train_path+file+'.npy' for file in train_filename ]
	x_train = np.array([ np.load(file) for file in train_filename ])

	train_label_dict = json.loads(open(train_label, 'r').read())
	y_train = [ label['caption'] for label in train_label_dict ]

	test_filename = open(test_id, 'r').read().splitlines()
	test_filename = [ test_path+file+'.npy' for file in test_filename ]
	x_test = np.array([ np.load(file) for file in test_filename ])

	test_label_dict = json.loads(open(test_label, 'r').read())
	y_test = [ label['caption'] for label in test_label_dict ]

	return (x_train, x_test), (y_train, y_test)

(x_train, x_test), (y_train, y_test) = load_data()
'''

def get_train_X():
	print("Loading training features . . .")

	train_path = './MLDS_hw2_1_data/training_data/'
	train_filename = open(train_path+'id.txt', 'r').read().split('\n')
	train_filename = [ train_path+'feat/'+file+'.npy' for file in train_filename ]
	train_filename.pop()
	#print(train_filename)

	arr = np.array([np.load(file) for file in train_filename])

	arr = 18*arr

#	arr.reshape((arr.shape[0], -1), order='F')

#	print(arr.shape)

#	print(arr)

	return arr

def get_train_Y():
	print("Loading training lables . . .")
	train_label_dict = json.loads(open('./MLDS_hw2_1_data/training_label.json', 'r').read())
	data = [ label['caption'] for label in train_label_dict ]
#	print(data)
#	print(":) ",data[0][0])

	caption = []
	for i in range(np.shape(data)[0]):#every video 1450
		per_video = []
		for strr in data[i]:
			temp = strr.split()
			per_video.append(temp)
		caption.append(per_video)
#	print(caption[0])
	dct = Dictionary(caption[0])
	for i in range(1,1450):
		dct.add_documents(caption[i])
#	print(dct)
	caption_num = []
	for i in range(1,1450): #every video 1450
		per_video = []
		r = np.shape(caption[i])[0]
		for j in range(80):#np.shape(caption[i])[0]):

			temp = dct.doc2idx(caption[i][j%r])

			per_video.append(temp)
		caption_num.append(per_video)
#	print(caption_num[0])
	return caption_num, dct


#Train

batch_size = 7

batches = get_train_X()
train_Y, dct = get_train_Y()

print("dct", dct)

'''helpers.random_sequences(
	length_from = 3, length_to = 8,
	vocab_lower = 2, vocab_upper = 10,
	batch_size = batch_size
	)'''

print("Train data ready!")


print("=================\n", batches)


def next_feed(_idx):
	b = batches[_idx]
	t = train_Y[_idx]


	encoder_inputs_, _ = helpers.batch(b)
	decoder_targets_, _ = helpers.batch(train_Y[_idx])
	decoder_inputs_, _ = helpers.batch(train_Y[_idx])
	
	_idx = _idx + 1

	print(encoder_inputs_)

	fdict = {
		encoder_inputs: encoder_inputs_,
		decoder_inputs: decoder_inputs_,
		decoder_targets: decoder_targets_
	}

#	print("dict:\n", dict)

	return fdict




loss_track = []

max_batches = 1000
batches_in_epochs = 1

_idx = 0
try:
	for batch in range(max_batches):
		fd = next_feed(_idx)
#		print("\n[train_op, loss]\n", train_op, "\n\nloss\n", loss)
#		print("\n\nfd\n", fd)
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
			#	tmp = (Dictionary(dct))(pred[0])
			#	print('\n            ', tmp )
				if i >= 2:
					break
			print()

except KeyboardInterrupt:
	print('training interrupted')

import matplotlib.pyplot as plt
plt.plot(loss_track)
plt.show()
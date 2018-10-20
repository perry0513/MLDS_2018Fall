import json
import numpy as np
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn_cell

# Load data
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

# build dictionary based on y_train & y_test
def build_dict(y_train, y_test):
	words = set()
	for vid in y_train:
		for sent in vid:
			add = [ w.lower().strip('".,') for w in sent.split(' ') ]
			words.update(add)
	for vid in y_test:
		for sent in vid:
			add = [ w.lower().strip('".,') for w in sent.split(' ') ]
			words.update(add)

	dict_in_one_hot = []
	for i in range(len(words)):
		one_hot = [0] * len(words)
		one_hot[i] = 1
		dict_in_one_hot.append(one_hot)

	dictionary = dict(zip(list(words), dict_in_one_hot))
	reverse_dict = dict(dictionary.values(), dictionary.key)
	print(dictionary)
	print(len(dictionary))
	return dictionary

build_dict(y_train, y_test)



encoder_units = 128
decoder_units = 128

def encoder():
	cell = rnn_cell.LSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

	return final_state[1]

def decoder():
	cell = rnn_cell.LSTMCell(n_hidden_units, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	outputs, final_state = tf.nn.dynamic_rnn(cell, final_state, time_major=False)










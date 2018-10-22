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
=========== to batch ============
shuffeled_data: (SHAPE: (1450, 80, 4096)
batch_size: shape of each batch
returns (SHAPE: (N, 80, BS, 4096))

def to_batch(shuffled_data, batch_size):
	four_dimension_batch_sized_input = []

	if import_size%batch_size : print("WARNING: not divisible")

	print("Separating into batch size . . .")

	for i in range(0, import_size, batch_size):
		temp = []
		if((i+batch_size-1) <= import_size):
			for b in range(batch_size):
				temp.append(shuffled_data[i+b])

			four_dimension_batch_sized_input.append(temp)

	# NOW:    (N, BS, 80FRAME, 4096)

	# NOW:    (N, 80Frame, BS, 4096)

	print("Separated!\nfour_dimension_batch_sized_input SHAPE: ", np.shape(four_dimension_batch_sized_input))

	return four_dimension_batch_sized_input
'''

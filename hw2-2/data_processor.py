import numpy as np

class DataProcessor():
	# Constant Parameters
	PAD = 0
	BOS = 1
	EOS = 2
	UNK = 3
	question_data_path = './MLDS_hw2_2_data/sel_conversation/question.txt'
	answer_data_path = './MLDS_hw2_2_data/sel_conversation/answer.txt'

	def __init__(self, min_count=20):
		self.min_count = min_count
		self.question_list = []
		self.answer_list = []
		self.encoder_inputs  = []
		self.decoder_inputs  = []
		self.decoder_targets = []
		self.encoder_inputs_length = []
		self.decoder_targets_length = []
		self.load_conversations()
		self.build_dictionary()
		self.encode_train_data()

	def get_dictionary(self):
		return self.idx2word_dictionary

	def get_batch(self, batch_size=10):
		total_data_num = len(self.question_list)
		zipped = list(zip(self.encoder_inputs, self.encoder_inputs_length, self.decoder_inputs, self.decoder_targets, self.decoder_targets_length))
		np.random.shuffle(zipped)
		shuffled_encoder_inputs, shuffled_encoder_inputs_length, shuffled_decoder_inputs, \
		shuffled_decoder_targets, shuffled_decoder_targets_length = [ np.array(tup) for tup in zip(*zipped) ]

		# zipped = list(zip(range(total_data_num), self.encoder_inputs, self.decoder_inputs, self.decoder_targets))
		# np.random.shuffle(zipped)
		# shuffled_idx, shuffled_encoder_inputs, shuffled_decoder_inputs, shuffled_decoder_targets = [ np.array(tup) for tup in zip(*zipped) ]
		# shuffled_encoder_inputs_length = [ len(self.question_list[idx]) for idx in shuffled_idx ]
		# shuffled_decoder_targets_length = [ len(self.answer_list[idx]) + 1 for idx in shuffled_idx ]

		# shuffled_encoder_inputs = [ np.array(v) for v in shuffled_encoder_inputs ]
		# shuffled_decoder_inputs = [ np.array(v) for v in shuffled_decoder_inputs ]
		# shuffled_decoder_targets = [ np.array(v) for v in shuffled_decoder_targets ]
		# shuffled_encoder_inputs_length = [ np.array(v) for v in shuffled_encoder_inputs_length ]
		# shuffled_decoder_targets_length = [ np.array(v) for v in shuffled_decoder_targets_length ]


		num_of_batch = total_data_num // batch_size
		batched_encoder_inputs  = np.split(np.array(shuffled_encoder_inputs[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_inputs  = np.split(np.array(shuffled_decoder_inputs[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_targets = np.split(np.array(shuffled_decoder_targets[: num_of_batch*batch_size ]), num_of_batch)
		batched_encoder_inputs_length  = np.split(np.array(shuffled_encoder_inputs_length[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_targets_length = np.split(np.array(shuffled_decoder_targets_length[: num_of_batch*batch_size ]), num_of_batch)

		return batched_encoder_inputs, batched_encoder_inputs_length, batched_decoder_inputs, batched_decoder_targets, batched_decoder_targets_length


	def encode_train_data(self):
		print('Encoding data . . .')
		for line in self.question_list:
			encoded_line = []
			for word in line:
				idx = self.word2idx_dictionary.get(word, False)
				encoded_line.append(idx if idx else self.UNK)
			encoded_line = encoded_line + [self.PAD]*(self.max_seq_length - len(encoded_line))
			self.encoder_inputs.append(np.array(encoded_line))

		for i, line in enumerate(self.answer_list):
			encoded_line = []
			for word in line:
				idx = self.word2idx_dictionary.get(word, False)
				encoded_line.append(idx if idx else self.UNK)
			self.decoder_inputs.append(np.array([self.BOS] + encoded_line + [self.PAD]*(self.max_seq_length - len(encoded_line))))
			self.decoder_targets.append(np.array(encoded_line + [self.PAD]*(self.max_seq_length - len(encoded_line)) + [self.EOS] ))
			
			if self.decoder_inputs[-1].shape[0] != 29 or self.decoder_targets[-1].shape[0] != 29:
				print("======== ERROR ========")
				print(self.max_seq_length)
				print(self.decoder_inputs[-1].shape[0])
				print(self.decoder_targets[-1].shape[0])



	def build_dictionary(self):
		print('Building dictionary . . .')
		# init
		# word2idx_dictionary format: {'word': word_idx}
		# idx2word_dictionary format: {word_idx: 'word'}
		self.word2idx_dictionary = {
			'<PAD>': self.PAD,
			'<BOS>': self.BOS,
			'<EOS>': self.EOS,
			'<UNK>': self.UNK
		}
		self.idx2word_dictionary = {
			self.PAD: '<PAD>',
			self.BOS: '<BOS>',
			self.EOS: '<EOS>',
			self.UNK: '<UNK>'
		}
		word_counts = {}
		current_dictionary_idx = 4
		self.max_seq_length = 0
		for line in self.question_list:
			length = len(line)
			self.encoder_inputs_length.append(length)
			if length > self.max_seq_length:
				self.max_seq_length = length
			for word in line:
				if self.word2idx_dictionary.get(word, False):
					continue
				count = word_counts.get(word, 0)
				if count == 0:
					word_counts.update({word: 1})
				elif count < self.min_count:
					word_counts.update({word: count+1})
				else:
					self.word2idx_dictionary.update({word: current_dictionary_idx})
					self.idx2word_dictionary.update({current_dictionary_idx: word})
					current_dictionary_idx += 1
		for line in self.answer_list:
			self.decoder_targets_length.append(len(line)+1)
		self.max_seq_length += 1



	def load_conversations(self):
		# load conversation for building dictionary
		print ("Loading conversations . . .")
		with open(self.question_data_path, 'r', encoding='utf8') as f:
			for line in f:
				line = line.rstrip()
				if line != "":
					self.question_list.append(line.split(' '))
		with open(self.answer_data_path, 'r', encoding='utf8') as f:
			for line in f:
				line = line.rstrip()
				if line != "":
					self.answer_list.append(line.split(' '))

### DEBUG CODE ###
# def check_same_length(arr):
# 	length = arr[0].shape[0]
# 	for v in arr:
# 		if v.shape[0] is not length:
# 			print("False")
# 			return
# 	print("True")

dp = DataProcessor()
print (len(dp.idx2word_dictionary))
batched_encoder_inputs, batched_encoder_inputs_length, batched_decoder_inputs, batched_decoder_targets, batched_decoder_targets_length = dp.get_batch(25)
print(np.array(batched_encoder_inputs).shape)
print(np.array(batched_encoder_inputs_length).shape)
print(np.array(batched_decoder_inputs).shape)
print(np.array(batched_decoder_targets).shape)
print(np.array(batched_decoder_targets_length).shape)

# check_same_length(batched_encoder_inputs[0])
# check_same_length(batched_encoder_inputs_length)
# check_same_length(batched_decoder_inputs[0])
# check_same_length(batched_decoder_targets[0])
# check_same_length(batched_decoder_targets_length)

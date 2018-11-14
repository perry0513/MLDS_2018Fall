import numpy as np

class DataProcessor():
	# Constant Parameters
	PAD = 0
	BOS = 1
	EOS = 2
	UNK = 3
	question_data_path = './MLDS_hw2_2_data/sel_conversation/question.txt'
	answer_data_path = './MDLS_hw2_2_data/sel_conversation/answer.txt'

	def __init__(self, min_count=20):
		self.min_count = min_count
		self.load_conversations()
		self.build_dictionary()

	def get_batch(self, batch_size=10):
		total_data_num = len(self.question_list)
		zipped = list(zip(self.encoder_inputs, self.encoder_inputs_length, self.decoder_inputs, self.decoder_targets, self.decoder_targets_length))
		np.random.shuffle(zipped)
		(shuffled_encoder_inputs, shuffled_encoder_inputs_length, shuffled_decoder_inputs,shuffled_decoder_targets, shuffled_decoder_targets_length) = [ np.array(tup) for tup in zip(*zipped) ]

		num_of_batch = total_data_num // batch_size
		batched_encoder_inputs  = np.split(np.array(shuffled_encoder_inputs[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_inputs = np.split(np.array(shuffled_decoder_inputs[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_targets = np.split(np.array(shuffled_decoder_targets[: num_of_batch*batch_size ]), num_of_batch)
		batched_encoder_inputs_length  = np.split(np.array(shuffled_encoder_inputs_length[: num_of_batch*batch_size ]), num_of_batch)
		batched_decoder_targets_length = np.split(np.array(shuffled_decoder_targets_length[: num_of_batch*batch_size ]), num_of_batch)

		return batched_encoder_inputs, batched_encoder_inputs_length, batched_decoder_inputs, batched_decoder_targets, batched_decoder_targets_length

	def encode_train_data(self):
		self.encoder_inputs  = []
		self.decoder_inputs  = []
		self.decoder_targets = []
		for line in self.question_list:
			encoded_line = []
			for word in line.split(' '):
				idx = self.word2idx_dictionary.get(word, False)
				if idx:
					encoded_line.append(idx)
				else:
					encoded_line.append(self.UNK)
			encoded_line = encoded_line + [self.PAD]*(self.max_seq_length-len(encoded_line))
			self.encoder_inputs.append(encoded_line)
		for line in self.answer_list:
			encoded_line = []
			for word in line.split(' '):
				idx = self.word2idx_dictionary.get(word, False)
				if idx:
					encoded_line.append(idx)
				else:
					encoded_line.append(self.UNK)
			self.decoder_targets.append(encoded_line + [self.PAD]*(self.max_seq_length-len(encoded_line)) + [self.EOS])
			self.decoder_inputs.append([self.BOS] + encoded_line + [self.PAD]*(self.max_seq_length-len(encoded_line)))
	def build_dictionary(self):
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
		self.encoder_inputs_length = []
		self.decoder_targets_length = []
		word_counts = {}
		current_dictionary_idx = 4
		self.max_seq_length = 0
		for line in self.question_list:
			split_line = line.split(' ')
			length = len(split_line)
			self.encoder_inputs_length.append(length)
			if length > self.max_seq_length:
				self.max_seq_length = length
			for word in split_line:
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
			self.decoder_targets_length.append(len(line.split(' '))+1)

	def load_conversations(self):
		# load conversation for building dictionary
		self.question_list = []
		self.answer_list = []
		with open(self.question_data_path, 'r', encoding='utf8') as f:
			for line in f:
				line = line.rstrip()
				if line != "":
					self.question_list.append(line)
		with open(self.answer_data_path, 'r', encoding='utf8') as f:
			for line in f:
				line = line.rstrip()
				if line != "":
					self.answer_list.append(line)
		print ("Loading conversations finished. ")


dp = DataProcessor()
print (len(dp.idx2word_dictionary))

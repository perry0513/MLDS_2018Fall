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
		self.load_conversations()
		self.build_dictionary()

	# def get_batch(self, batch_size=10):

	def build_train_data(self):
		question_list = []
		answer_list = []
		for conversation in self.conversation_list:
			is_question = True
			for line in conversation:
				# unfinished
				print ("hello, world")
			
	def build_dictionary(self):
		# init
		# word2idx_dictionary format: {'word': word_idx}
		# idx2word_dictionary format: {word_idx: 'word'}
		# data_encoded_by_idx format: videos[captoions[words[]]]
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
		for line in self.question_list:
			for word in line.split(' '):
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

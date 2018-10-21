import numpy as np
import json
from gensim.corpora import Dictionary
import store_data as sd

import_size = 1450

# Shuffles sentence(1490,?), video(1450,80,4096)
# Then separates video into (N, BS, 80, 4096)
# NO!!!!! Transposes video data into (N, 80, BS, 4096)

def shuffle_and_zip(sentences, video, BS):
	print("Shuffling data . . .")

	idx = np.arange(import_size)

	zipped = list(zip(idx,sentences))
	np.random.shuffle(zipped)
	idx, shuffled_sentences = [ np.array(t) for t in zip(*zipped) ]

	BS_mod = import_size//BS * BS

	shuffled_data = [ np.array(video[idx[i]]) for i in range(import_size) ]
	shuffled_data = np.asarray(shuffled_data)
	shuffled_data = np.split(shuffled_data[:BS_mod],  import_size // BS, axis = 0)

#	shuffled_data_T = np.transpose(shuffled_data, (0,2,1,3))


	print("(N, batch_size, frame, pixel) = ", np.shape(shuffled_data))

	return shuffled_data, shuffled_sentences


'''
### TESTING DEBUG
_, s = sd.get_sentence()
v = sd.get_video()
#print("????", s)
#print("\n", np.shape(s))
shuffle_and_zip(s, v, 7)
'''

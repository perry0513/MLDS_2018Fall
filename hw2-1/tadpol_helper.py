import numpy as np
import json
import store_data as sd

import_size = 1450

# Shuffles sentence(1450,?), video(1450,80,4096)
# Then separates video into (N, BS, 80, 4096)
# (X) Transposes video data into (N, 80, BS, 4096)

# Returns shuffled and batched data & sentences

def shuffle_and_zip(groups_of_sentences, video, BS):
	print("Shuffling video . . .")

	idx = np.arange(import_size)

	rand_choose_sentences = [ sentences[np.random.choice(len(sentences))] for sentences in groups_of_sentences ]

	zipped = list(zip(idx,rand_choose_sentences))
	np.random.shuffle(zipped)
	idx, shuffled_sentence = [ np.array(t) for t in zip(*zipped) ]

	BS_mod = import_size//BS * BS

	shuffled_video = [ np.array(video[idx[i]]) for i in range(import_size) ]
	shuffled_video = np.asarray(shuffled_video)
	shuffled_video = np.split(shuffled_video[:BS_mod],  import_size // BS, axis = 0)

	shuffled_sentence = np.asarray(shuffled_sentence)
	shuffled_sentence = np.split(shuffled_sentence[:BS_mod],  import_size // BS, axis = 0)


#	shuffled_video_T = np.transpose(shuffled_video, (0,2,1,3))


	print("(N, batch_size, frame, pixel) = ", np.shape(shuffled_sentence))

	return shuffled_video, np.array(shuffled_sentence)



### TESTING DEBUG
_, s = sd.get_sentence()
v = sd.get_video()
#print("????", s)
#print("\n", np.shape(s))
video, sent = shuffle_and_zip(s, v, 7)

# print(sent)
print(sent.shape)


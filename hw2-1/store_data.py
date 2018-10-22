import json
import numpy as np

'''
=========== get_video ============
returns SHAPE(1450, 80, 4096)
'''
def get_video():
    print("Loading training features . . .")

    train_path = './MLDS_hw2_1_data/training_data/'
    train_filename = open(train_path+'id.txt', 'r').read().splitlines()
    train_filename = [ train_path+'feat/'+file+'.npy' for file in train_filename ]
    #print(train_filename)

    arr = np.array([np.load(file) for file in train_filename])

    return arr


def get_sentence():
    #### Load Data ####
    print("Loading training lables . . .")
    train_label_dict = json.loads(open('./MLDS_hw2_1_data/training_label.json', 'r').read())
    data = [ label['caption'] for label in train_label_dict ]

    #### Store Data ####
    # init
    # word2idx_dictionary format: {'word': word_idx}
    # idx2word_dictionary format: {word_idx: 'word'}
    # data_encoded_by_idx format: videos[captoions[words[]]]
    PAD = 0
    BOS = 1
    EOS = 2
    word2idx_dictionary = {
        '<PAD>': PAD,
        '<BOS>': BOS,
        '<EOS>': EOS
    }
    idx2word_dictionary = {
        PAD: '<PAD>',
        BOS: '<BOS>',
        EOS: '<EOS>'
    }
    data_encoded_by_idx = []
    current_idx = 3
    max_caption_length = 0

    # create the dictionaries and encode data by word2idx
    for videos in data:
        videos_encoded_by_idx = []
        for captions in videos:
<<<<<<< HEAD
            captions_encoded_by_idx = [1]
=======
            captions_encoded_by_idx = [BOS]
>>>>>>> 3b80e8463bf1fcf3928fcbfd905444144a9475d1
            for word in captions.split():
                # clear the ",. in the words
                word = word.lower().strip('",.')

                # add the word to dictionaries if not existed
                if not word2idx_dictionary.get(word):
                    word2idx_dictionary.update({word: current_idx})
                    idx2word_dictionary.update({current_idx: word})
                    current_idx += 1
                
                captions_encoded_by_idx.append(word2idx_dictionary.get(word))
            
            if max_caption_length < len(captions_encoded_by_idx):
                max_caption_length = len(captions_encoded_by_idx)
            
            videos_encoded_by_idx.append(captions_encoded_by_idx)
        data_encoded_by_idx.append(videos_encoded_by_idx)

    # fill the list with <PAD> behind
    for videos in data_encoded_by_idx:
        for captions in videos:
            captions += [PAD] * (max_caption_length-len(captions))

    del word2idx_dictionary

    return idx2word_dictionary, data_encoded_by_idx

import json

def load_and_store_data():
    #### Load Data ####
    print("Loading training lables . . .")
    train_label_dict = json.loads(open('./MLDS_hw2_1_data/training_label.json', 'r').read())
    data = [ label['caption'] for label in train_label_dict ]

    #### End of Load Data ####

    #### Store Data ####
    # read the data and create two dictionary 
    # word2idx_dictionary format: {'word': word_idx}
    # idx2word_dictionary format: {word_idx: 'word'}
    # clean the word2idx_dictionary in the end

    # init
    word2idx_dictionary = {
        "PAD": 0,
        "BOS": 1,
        "EOS": 2
    }
    idx2word_dictionary = {
        0: "PAD",
        1: "BOS",
        2: "EOS"
    }
    data_encoded_by_idx = []
    idx = 3
    # walk through every video
    for videos in data:
        videos_encoded_by_idx = []
        for captions in videos:
            captions_encoded_by_idx = []
            for word in captions.split():
                word = word.lower().strip('",.')
                # add the word to dictionaries if not existed
                if (not word2idx_dictionary.get(word)):
                    word2idx_dictionary.update({word: idx})
                    idx2word_dictionary.update({idx: word})
                    idx += 1
                
                captions_encoded_by_idx.append(word2idx_dictionary.get(word))
            videos_encoded_by_idx.append(captions_encoded_by_idx)
        data_encoded_by_idx.append(videos_encoded_by_idx)
    
    #### End of Store Data ####
    
    return idx2word_dictionary, data_encoded_by_idx

if __name__ == "__main__":
    print load_and_store_data()[1][0]
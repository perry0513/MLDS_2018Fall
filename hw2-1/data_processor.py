import json
import numpy as np

class DataProcessor():
    # Constant Parameters
    TRAIN = 0
    TEST = 1
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, mode, max_caption_length=0, current_dictionary_idx=0):
        self.mode = mode
        self.max_caption_length = max_caption_length
        self.current_dictionary_idx = current_dictionary_idx
        self.load_videos()
        self.training_label = self.load_label('train')
        self.testing_label = self.load_label('test')
        self.set_dictionary_and_captions()
        self.set_decoder_inputs()
        self.set_decoder_targets()


    #### Access Functions ####
    def get_videos(self):
        print('Loading videos . . .')
        return np.array(self.feat)
    
    def get_dictionary(self):
        print('Loading dicitonary . . .')
        return self.idx2word_dictionary

    def get_decoder_inputs(self):
        print('Loading decoder inputs . . .')
        return np.array(self.decoder_inputs)

    def get_decoder_targets(self):
        print('Loading decoder targets . . .')
        return np.array(self.decoder_targets)
    
    #### Process Functions ####
    # returns SHAPE(1450, 80, 4096)
    def load_videos(self):
        print('Loading ' + self.mode + 'ing features . . .')

        path = './MLDS_hw2_1_data/' + self.mode + 'ing_data/'
        filename = open(path+'id.txt', 'r').read().splitlines()
        filename = [ path+'feat/'+file+'.npy' for file in filename ]

        self.feat = np.array([np.load(file) for file in filename])

    def load_label(self, mode):
        print('Loading ' + mode + 'ing lables . . .')

        label_dict = json.loads(open('./MLDS_hw2_1_data/' + mode + 'ing_label.json', 'r').read())
        label = [ labels['caption'] for labels in label_dict ]

        return label

    # save data in self
    def set_dictionary_and_captions(self):
        # init
        # word2idx_dictionary format: {'word': word_idx}
        # idx2word_dictionary format: {word_idx: 'word'}
        # data_encoded_by_idx format: videos[captoions[words[]]]
        self.word2idx_dictionary = {
            '<PAD>': self.PAD,
            '<BOS>': self.BOS,
            '<EOS>': self.EOS
        }
        self.idx2word_dictionary = {
            self.PAD: '<PAD>',
            self.BOS: '<BOS>',
            self.EOS: '<EOS>'
        }
        self.current_dictionary_idx = 3
        self.training_data_encoded_by_idx = []
        self.testing_data_encoded_by_idx = []

        # create the dictionaries and encode data by word2idx
        self.append_dictionary_and_captions(self.TRAIN)
        self.append_dictionary_and_captions(self.TEST)

        del self.word2idx_dictionary
        del self.training_label
        del self.testing_label

    def append_dictionary_and_captions(self, mode):
        for videos in (self.training_label if mode == self.TRAIN else self.testing_label):
            videos_encoded_by_idx = []
            for captions in videos:
                captions_encoded_by_idx = []
                for word in captions.split():
                    # clear the ",. in the words
                    word = word.lower().strip('",.')

                    # add the word to dictionaries if not existed
                    if not self.word2idx_dictionary.get(word):
                        self.word2idx_dictionary.update({word: self.current_dictionary_idx})
                        self.idx2word_dictionary.update({self.current_dictionary_idx: word})
                        self.current_dictionary_idx += 1
                    
                    captions_encoded_by_idx.append(self.word2idx_dictionary.get(word))
                
                if self.max_caption_length < len(captions_encoded_by_idx):
                    self.max_caption_length = len(captions_encoded_by_idx)
                
                videos_encoded_by_idx.append(captions_encoded_by_idx)
            
            self.training_data_encoded_by_idx.append(videos_encoded_by_idx) if mode == self.TRAIN else self.testing_data_encoded_by_idx.append(videos_encoded_by_idx)


    def set_decoder_inputs(self):
        self.decoder_inputs = []
        for videos in (self.training_data_encoded_by_idx if self.mode == 'train' else self.testing_data_encoded_by_idx):
            self.decoder_inputs.append( np.array([ np.array([self.BOS] + captions + [self.PAD]*(self.max_caption_length-len(captions))) for captions in videos ] ) )

    def set_decoder_targets(self):
        self.decoder_targets = []
        for videos in (self.training_data_encoded_by_idx if self.mode == 'train' else self.testing_data_encoded_by_idx):
            self.decoder_targets.append( np.array([ np.array(captions + [self.EOS]) for captions in videos ] ) )


dataprocessor = DataProcessor('test')
print(dataprocessor.get_videos().shape)
print(dataprocessor.get_dictionary())
print(dataprocessor.get_decoder_inputs())
print(dataprocessor.get_decoder_targets().shape)
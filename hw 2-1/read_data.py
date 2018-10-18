import numpy as np

train_path = './MLDS_hw2_1_data/training_data/'
train_filename = open(train_path+'id.txt', 'r').read().split('\n')
train_filename = [ train_path+'feat/'+file+'.npy' for file in train_filename ]
train_filename.pop()
print(train_filename)

arr = np.array([np.load(file) for file in train_filename])

print(arr.shape)
import numpy as np
import os
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class DataProcessor():
	def __init__(self, image_num=35916):
		# Constant Parameters
		self.image_num = image_num
		self.image_path = './AnimeDataset/extra_data/images/'
		self.img_npy_path = './AnimeDataset/filtered_data.npy'
		self.tag_path = './AnimeDataset/filtered_tags.csv'
# 		self.image_path = './AnimeDataset/extra_data/images/'
# 		self.npy_path = './AnimeDataset/extra_faces.npy'
		self.image_list = []
		self.load_data()

	def load_data(self):
		print ('Loading image data . . .', end='\r')
		
		if os.path.isfile(self.img_npy_path):
			self.image_list = np.load(self.img_npy_path)
		else:
			image_list = []
			for i in tqdm(range(self.image_num)):
# 				print('Loading ', i+1, ' . . .')
				image_bgr = cv2.resize(cv2.imread(self.image_path+str(i)+".jpg"), (64, 64))
				image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
				image_list.append(image_rgb)

			image_list = np.array(image_list)

			# normalize (-1 ,1)
			self.image_list = image_list / 128 - 1
			# save as .npy file
			np.save(self.img_npy_path, self.image_list)
			
		print('Image loaded')
		print('Loading tags . . .', end='\r')
		
		tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'red eyes',
					'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 'gray eyes', 'black eyes', 'blue eyes',
					'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes']
		tag_csv = open(self.tag_path, 'r').readlines()
		
		label_list = []
		for i in range(len(tag_csv)):
			id, tag = tag_csv[i].split(',')
			label = np.zeros(len(tag_dict),dtype=float)
			for j in range(len(tag_dict)):
				if tag_dict[j] in tag:
					label[j] = 1.
# 			print(label)
			
			label_list.append(label)
	
		self.label_list = np.array(label_list)
		
		print('Label loaded')
						
			

		


	def get_batch(self, batch_size, batch_needed):
		index = np.arange(self.image_num)
		np.random.shuffle(index)
		images = np.array([ self.image_list[idx] for idx in index ])
		labels = np.array([ self.label_list[idx] for idx in index ])
		wrong_labels = labels.copy()
		np.random.shuffle(wrong_labels)
		
		batch_num = self.image_num // batch_size
		batched_images = np.split(images[: batch_size*batch_num ], batch_num)
		batched_labels = np.split(labels[: batch_size*batch_num ], batch_num)
		batched_wrong_labels = np.split(wrong_labels[: batch_size*batch_num ], batch_num)

		return batched_images, batched_labels, batched_wrong_labels

# dp = DataProcessor()
# a, b, c = dp.get_batch(100, 3)
# # # a, b = dp.get_batch(100, 3)
# # # a, b = dp.get_batch(100, 3)
# # # print(np.array(a).shape, np.array(b).shape)
# for i in range(len(b[0])):
# 	print(i, end=' ')
# 	for j in range(22):
# 		print(b[0][i][j], end=' ')
# 	print()
# # print(a[0][0][0][0])
# print(a[1][0][0][0])
# print(a[2][0][0][0])

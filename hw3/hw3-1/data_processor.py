import numpy as np
import os
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class DataProcessor():
	def __init__(self, image_num=36739):
		# Constant Parameters
		self.image_num = image_num
		self.image_path = './AnimeDataset/extra_data/images/'
		self.npy_path = './AnimeDataset/extra_faces.npy'
		self.image_list = []
		self.load_data()

	def load_data(self):
		print ('Loading image data . . .')
		
		if os.path.isfile(self.npy_path):
			self.image_list = np.load(self.npy_path)
		else:
			image_list = []
			for i in tqdm(range(self.image_num)):
				image_bgr = cv2.resize(cv2.imread(self.image_path+str(i)+".jpg"), (64, 64))
				image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
				image_list.append(image_rgb)

			image_list = np.array(image_list)

			# normalize (-1 ,1)
			self.image_list = image_list / 128 - 1
			# save as .npy file
			np.save(self.npy_path, self.image_list)


	def get_batch(self, batch_size, batch_needed):
		index = np.arange(self.image_num)
		np.random.shuffle(index)
		images = np.array([ self.image_list[idx] for idx in index ])
		batch_num = self.image_num // batch_size

		batched_img = np.split(images[: batch_size*batch_num ], batch_num)
		# batched_label = np.array([ np.ones(batch_size) for i in range(batch_num) ])

		return batched_img # batched_label

# dp = DataProcessor()
# a = dp.get_batch(100, 3)
# print(a[0].shape)
# print(a[0][0][0][0])
# print(a[1][0][0][0])
# print(a[2][0][0][0])

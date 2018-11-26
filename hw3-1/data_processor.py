import numpy as np
import cv2

class DataProcessor():
	def __init__(self, image_num=33431):
		# Constant Parameters
		self.image_num = image_num
		self.image_path = "./AnimeDataset/faces/"
		self.image_list = []
		self.load_data()

	def load_data(self):
		print ("Loading image data . . .")
		image_list = []
		for i in range(self.image_num):
			image_list.append(cv2.resize(cv2.imread(self.image_path+str(i)+".jpg"), (64, 64)))
		image_list = np.array(image_list)

		# normalize
		self.image_list = image_list / 256
		print (self.image_list.shape)
		print (self.image_list[0])


	def get_batch(self, batch_size, d_iter):
		np.random.shuffle(self.image_list)
		batch_num = self.image_num // batch_size

		batched_img = np.split(self.image_list[: batch_size*batch_num ], batch_num)
		# batched_label = np.array([ np.ones(batch_size) for i in range(batch_num) ])

		return batched_img[:d_iter] # batched_label

# dp = DataProcessor()
# dp.load_data()
# a = dp.get_batch(100)
# print(a[0].shape)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision
# from keras.datasets import mnist
# from keras.utils import np_utils

torch.manual_seed(1) #reproducible

# def load_data(train_size):
# 	(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 	x_train = x_train[0:train_size]
# 	y_train = y_train[0:train_size]
# 	x_train = x_train.reshape(train_size, 28*28)
# 	x_test = x_test.reshape(x_test.shape[0], 28*28)
# 	x_train = x_train.astype('float32')
# 	x_test = x_test.astype('float32')

# 	y_train = np_utils.to_categorical(y_train, 10)
# 	y_test = np_utils.to_categorical(y_test, 10)
# 	x_train = x_train / 255
# 	x_test = x_test / 255

# 	return (x_train, y_train) , (x_test, y_test)

# Config
TRAIN_SIZE = 10000
TEST_SIZE = 2000
BATCH_SIZE = 100
EPOCHS = 100

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 10)
		self.fc2 = nn.Linear(10, 10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

model = Net()
if torch.cuda.is_available():
	model.cuda()
print(model)

# pack data
# (x_train, y_train) , (x_test, y_test) = load_data(TRAIN_SIZE)
# train_dataset = utils.TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train))
# test_dataset = utils.TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test))
# train_loader  = utils.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
# test_loader   = utils.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# pack data
train_data = dset.MNIST(root='./MNIST-data/', train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = dset.MNIST(root='./MNIST-data/', train=False, transform=torchvision.transforms.ToTensor())
train_loader = utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = utils.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:]/255.
# label_y = test_data.test_labels[:].view(-1,1)
# test_y = torch.zeros(len(label_y), 10).scatter_(1,label_y,1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

if torch.cuda.is_available:
	criterion.cuda()

# train
loss = 0
loss_hist = []
total_step = len(train_loader)
params_hist = torch.Tensor()
if torch.cuda.is_available:
	params_hist = params_hist.cuda()
for epoch in range(EPOCHS):
    # collect params
	if epoch % 3 == 0:
		params = torch.Tensor()
		if torch.cuda.is_available:
			params = params.cuda()
		for p in model.parameters():
			params = torch.cat((params,p.view(1,-1)),1)
		params_hist = torch.cat((params_hist,params),0)
	for i, (x, y) in enumerate(train_loader):
		# label_y = y.view(-1,1)
		# print (label_y)
		# onehot_y = torch.zeros(len(y), 10).scatter_(1,label_y,1)
		# print (onehot_y)
		if torch.cuda.is_available:
			x,y = x.cuda(), y.cuda()
		optimizer.zero_grad()
		output = model(x.view(BATCH_SIZE,-1))
		loss = criterion(output, y.long())
		loss.backward()
		optimizer.step()
		if (i+1) % 50 == 0:
			print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.4f}'.format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
		loss_hist.append(loss)
# PCA
# print (params_hist.shape)
# mean = params_hist.mean(dim=1, keepdim=True)
# std = params_hist.std(dim=1, keepdim=True)
# params_hist_norm = (params_hist - mean) / std
# cov_mat = np.cov(params_hist_norm.cpu().detach().numpy())
# print (cov_mat.shape)

U,S,V = torch.svd(torch.transpose(params_hist,0,1))
print(params_hist.shape)
print(U.shape)
C = torch.mm(params_hist,U[:,:2])
print (C.shape)

# test
with torch.no_grad():
	total_loss = 0
	for x, y in test_loader:
		if torch.cuda.is_available:
			x,y = x.cuda(),y.cuda()    		
		output = model(x.view(BATCH_SIZE,-1))
		loss = criterion(output, y)
		total_loss = total_loss + loss.cpu().numpy() / total_step
	print('Test loss: ', total_loss)

# plot
with np.errstate(invalid='ignore', divide='ignore') and torch.no_grad():
	C = C.transpose(0,1)
	x_plot = C[0].cpu().detach().numpy()
	y_plot = C[1].cpu().detach().numpy()
	plt.scatter(x_plot,y_plot)
	plt.show()
import numpy as np
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
from random import random
from sklearn.manifold import TSNE


def func(arr):
	# return 0.5*np.sin(arr)
	# return np.sin(5*np.pi*(arr+0.1))/(5*np.pi*(arr+0.1))+10
	return np.exp(np.sin(40*arr))*np.log(arr+1) / 10
	# return np.sin(3*np.pi*arr)+np.sin(4*np.pi*arr)

# hyper parameters
TRAIN_SIZE = 10000
BATCH_SIZE = 1000
EPOCHS = 30
SAMPLE = 200
NOISE_STD = 1e-4
EPOCH_LFBGS = 1	# dangerous if too big
MAX_GRAD = 5.0


loss_plot, param_plot = [], []

def test(m, total_step):
	with torch.no_grad():
		total_loss = 0
		for x, y in test_loader:
			output = m(x)
			loss = criterion(output, y)
			total_loss = total_loss + loss.numpy() / total_step
		return total_loss


model = nn.Sequential(
			nn.Linear(1, 4),
			nn.ReLU(),
			nn.Linear(4, 9),
			nn.ReLU(),
			nn.Linear(9,16),
			nn.ReLU(),
			nn.Linear(16,9),
			nn.ReLU(),
			nn.Linear(9, 4),
			nn.ReLU(),
			nn.Linear(4, 1)
		)

# pack data
x_train, x_test = torch.rand(TRAIN_SIZE, 1), torch.rand(TRAIN_SIZE, 1)
train_dataset = utils.TensorDataset(x_train, func(x_train))
test_dataset  = utils.TensorDataset(x_test, func(x_test))
train_loader  = utils.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader   = utils.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# train
loss, grad_norm = 0, 0
loss_hist, norm_hist = [], []
total_step = len(train_loader)
for epoch in range(EPOCHS):
	for i, (x, y) in enumerate(train_loader):
		optimizer.zero_grad()
		output = model(x)
		# objective function = loss function
		loss = criterion(output, y)
		# compute gradient norm
		g = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
		g_sq = sum([grad.norm() ** 2 for grad in g])
		norm = g_sq.cpu().data.sqrt()

		loss.backward()
		optimizer.step()

		if (i+1) % 10 == 0:
			print ('Epoch [{}/{}] |\tStep [{}/{}] |\t\tLoss: {:.6f} |\t grad_norm: {:.5f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))			

			
	loss_hist.append(loss)
	norm_hist.append(norm)


optimizer = optim.LBFGS(model.parameters(), lr=1)
params_hist = torch.Tensor()
params_loss_hist = []
params_hist_around = torch.Tensor()
params_loss_hist_around = []
for epoch in range(EPOCH_LFBGS):
	for i, (x, y) in enumerate(train_loader):
		def closure():
			output, loss = 0, 0
			optimizer.zero_grad()
			output = model(x)
			if math.isnan(output[0]):
				print('WONDERFUL:)')
				sys.exit()
			loss = criterion(output, y)
			loss.backward()

			# check if grad is too large, limit it in [ -MAX_GRAD, MAX_GRAD ]
			for layer in model:
				if type(layer) == nn.Linear:
					for r in range(len(layer.weight.grad)):
						for i in range(len(layer.weight.grad[r])):
							if abs(layer.weight.grad[r][i]) > MAX_GRAD:
								layer.weight.grad[r][i] = (MAX_GRAD if layer.weight.grad[r][i] > MAX_GRAD else -MAX_GRAD)*random()

					for i in range(len(layer.bias.grad)):
						if abs(layer.bias.grad[i]) > MAX_GRAD:
							layer.bias.grad[i] = (MAX_GRAD if layer.bias.grad[i] > MAX_GRAD else -MAX_GRAD)*random()

			return loss

		loss = optimizer.step(closure)

		# record model and loss
		params = torch.Tensor()
		for layer in model:
			if type(layer) == nn.Linear:
				params = torch.cat((params, layer.weight.view(1,-1)),1)
				params = torch.cat((params, layer.bias.view(1,-1)),1)
		params_hist = torch.cat((params_hist, params),0)
		params_loss_hist.append(loss)


		# sample points around original model and calculate loss
		n = torch.distributions.Normal(torch.Tensor([0.]), torch.Tensor([NOISE_STD]))
		def add_noise(m):
			if type(m) == nn.Linear:
				# n.sample will add an extra dim(?), so squeeze
				m.weight.data.add_(n.sample(m.weight.data.shape).squeeze(2))
				m.bias.data.add_(n.sample(m.bias.data.shape).squeeze(1))

		for i in tqdm(range(SAMPLE)):
			# deep copy model
			model_cp = copy.deepcopy(model)
			# add noise to copied model
			model_cp.apply(add_noise)

			m_loss = test(model_cp, total_step)

			params = torch.Tensor()

			for layer in model:
				if type(layer) == nn.Linear:
					params = torch.cat((params, layer.weight.view(1,-1)),1)
					params = torch.cat((params, layer.bias.view(1,-1)),1)
			params_hist_around = torch.cat((params_hist_around, params),0)
			params_loss_hist_around.append(m_loss)

		print ('Epoch [{}/{}] |\tStep [{}/{}] |\t\tLoss: {:.6f}' 
               .format(epoch+1, EPOCH_LFBGS, i+1, total_step, loss.item()))
			

		

	# for p in model.parameters():
	# 	params = torch.cat((params,p.view(1,-1)),1)
	# params_hist = torch.cat((params_hist,params),0)
	# params_loss_hist.append(loss)

		# params = torch.Tensor()
		# for p in model_cp.parameters():
		# 	params = torch.cat((params,p.view(1,-1)),1)
		# params_hist_around = torch.cat((params_hist,params),0)
		# params_loss_hist_around.append(loss)
	loss_hist.append(loss)

print(loss_hist[-10:])



# Plot
X = TSNE(n_components=2).fit_transform(params_hist.detach())
X = np.transpose(X)
x_plot = X[0]
y_plot = X[1]
plt.scatter(x_plot,y_plot,s=3)
for i,(j,k) in enumerate(zip(x_plot,y_plot)):
	plt.annotate("%.6f"%params_loss_hist[i],xy=(j,k),fontsize=8,color='b')
plt.show()

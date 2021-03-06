import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import math
from sklearn.manifold import TSNE

def func(arr):
	# return 0.5*np.sin(arr)
	# return np.sin(5*np.pi*(arr+0.1))/(5*np.pi*(arr+0.1))+10
	return np.exp(np.sin(40*arr))*np.log(arr+1) / 10
	# return np.sin(3*np.pi*arr)+np.sin(4*np.pi*arr)

# hyper parameters
TRAIN_SIZE = 10000
BATCH_SIZE = 100
EPOCHS = 20
SAMPLE = 200
NOISE_STD = 1e-4
EXPERIMENTS = 1
EPOCH_LFBGS = 10 	# dangerous if too big
MAX_GRAD = 5.0

min_ratio, min_loss = [], []

for exp in range(EXPERIMENTS):
	model = nn.Sequential(
				nn.Linear(1, 4),
				nn.ReLU(),
				nn.Linear(4, 9),
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

			if (i+1) % 100 == 0:
				print ('Data [{}/{}] |\tEpoch [{}/{}] |\tStep [{}/{}] |\t\tLoss: {:.6f} |\t grad_norm: {:.5f}' 
	                   .format(exp+1, EXPERIMENTS, epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))			

			# print ('Data [{}/{}] |\tEpoch [{}/{}] |\tStep [{}/{}] |\t\tLoss: {:.6f} |\t grad_norm: {:.5f}' 
                   # .format(exp+1, EXPERIMENTS, epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))
				
		loss_hist.append(loss)
		norm_hist.append(norm)

	def plot():
		with np.errstate(invalid='ignore', divide='ignore') and torch.no_grad():
			x_plot = np.arange(0.0, 1.0, 0.00001, dtype='float32')
			y_plot = func(x_plot)
			plt.plot(x_plot, model(torch.from_numpy(x_plot).reshape(len(x_plot), 1)).squeeze().numpy(), label='model')
			plt.plot(x_plot, y_plot, label='function')
			plt.title('sim func')
			plt.xlabel('x')
			plt.ylabel('y')	
			plt.legend()
			plt.show()

		plt.subplot(2,1,1)
		plt.plot(loss_hist)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		# plt.yscale('log')
		plt.subplot(2,1,2)
		plt.plot(norm_hist)
		plt.xlabel('epoch')
		plt.ylabel('grad')
		# plt.yscale('log')
		plt.show()
	# plot()


	optimizer = optim.LBFGS(model.parameters(), lr=0.1)
	params_hist = torch.Tensor()
	params_loss_hist = []
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

				for layer in model:
					if type(layer) == nn.Linear:
						WOOOOOOOOOOOOOOOW_w, WOOOOOOOOOOOOOOOW_b = False, False
						weight_printed, bias_printed = False, False
						for r in range(len(layer.weight.grad)):
							for i in range(len(layer.weight.grad[r])):
								if abs(layer.weight.grad[r][i]) > MAX_GRAD:
									if not weight_printed: 
										print('OLD WEIGHT\n',layer.weight.grad)
										weight_printed = True
									layer.weight.grad[r][i] = MAX_GRAD if layer.weight.grad[r][i] > 5 else -MAX_GRAD
									# layer.weight.grad = torch.zeros_like(layer.weight)
									WOOOOOOOOOOOOOOOW_w = True
								# if elem > MAX_GRAD:
								# 	print('OLD WEIGHT\n',layer.weight.grad)
								# 	elem = MAX_GRAD
								# 	WOOOOOOOOOOOOOOOW_w = True
								# elif elem < -MAX_GRAD:
								# 	print('OLD WEIGHT\n',layer.weight.grad)
								# 	elem = -MAX_GRAD
								# 	WOOOOOOOOOOOOOOOW_w = True
						for i in range(len(layer.bias.grad)):
							if abs(layer.bias.grad[i]) > MAX_GRAD:
								layer.bias.grad[i] = MAX_GRAD if layer.bias.grad[i] > 5 else -MAX_GRAD
								# layer.bias.grad = torch.zeros_like(layer.bias)
								WOOOOOOOOOOOOOOOW_b = True
								if not bias_printed:
									print('OLD BIAS\n',layer.bias.grad)
									bias_printed = True
							# if elem > MAX_GRAD:
							# 	print('OLD BIAS\n',layer.bias.grad)
							# 	elem = MAX_GRAD
							# 	WOOOOOOOOOOOOOOOW_b = True
							# elif elem < -MAX_GRAD:
							# 	print('OLD BIAS\n',layer.bias.grad)
							# 	elem = -MAX_GRAD
							# 	WOOOOOOOOOOOOOOOW_b = True
						if WOOOOOOOOOOOOOOOW_w:
							print('NEW WEIGHT_GRAD\n', layer.weight.grad)
						if WOOOOOOOOOOOOOOOW_b:
							print('NEW BIAS_GRAD\n', layer.bias.grad)
				return loss

			loss = optimizer.step(closure)
			params = torch.Tensor()
			for p in model.parameters():
				params = torch.cat((params,p.view(1,-1)),1)
			params_hist = torch.cat((params_hist,params),0)
			params_loss_hist.append(loss)

			print ('Data [{}/{}] |\tEpoch [{}/{}] |\tStep [{}/{}] |\t\tLoss: {:.6f}' 
	               .format(exp+1, EXPERIMENTS, epoch+1, EPOCH_LFBGS, i+1, total_step, loss.item()))
				
		loss_hist.append(loss)

	print(loss_hist[-10:])
	# plot()
	# return test loss
	def test(m):
		with torch.no_grad():
			total_loss = 0
			for x, y in test_loader:
				output = m(x)
				loss = criterion(output, y)
				total_loss = total_loss + loss.numpy() / total_step
			return total_loss

	test_loss = test(model)
	print("Test loss: ", test_loss)



	# Sample & determine minimal ratio
	# creates normal dist. with mean = 0 std = 1e-4
	n = torch.distributions.Normal(torch.Tensor([0.]), torch.Tensor([NOISE_STD]))

	# add noise to parameters
	def add_noise(m):
		if type(m) == nn.Linear:
			# n.sample will add an extra dim(?), so squeeze
			m.weight.data.add_(n.sample(m.weight.data.shape).squeeze(2))
			m.bias.data.add_(n.sample(m.bias.data.shape).squeeze(1))

	min_count = 0
	for i in tqdm(range(SAMPLE)):
		# deep copy model
		model_cp = copy.deepcopy(model)
		# add noise to copied model
		model_cp.apply(add_noise)

		m_loss = test(model_cp)
		if m_loss > test_loss: 
			min_count = min_count + 1

	print('(min ratio, loss, eq_count): ({},{})'.format(min_count/SAMPLE, test_loss))
	min_ratio.append(min_count/SAMPLE)
	min_loss.append(test_loss)
	# plot()


print (params_hist.shape)
X = TSNE(n_components=2).fit_transform(params_hist.detach())
X = np.transpose(X)
x_plot = X[0]
y_plot = X[1]
plt.scatter(x_plot,y_plot,s=3)
for i,(j,k) in enumerate(zip(x_plot,y_plot)):
	plt.annotate("%.6f"%params_loss_hist[i],xy=(j,k),fontsize=8,color='b')
plt.show()

# plt.plot(min_ratio, min_loss, 'ro')
# plt.xlim(-0.01, max(min_ratio)+0.02)
# plt.ylim(-0.01, 0.2)
# # plt.margins(0.5, tight=None)
# plt.title('STD of noise: '+ str(NOISE_STD))
# plt.xlabel('min ratio')
# plt.ylabel('loss')
# plt.show()
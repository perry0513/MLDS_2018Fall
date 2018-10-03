import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(1, 4)
		self.fc2 = nn.Linear(4, 9)
		self.fc3 = nn.Linear(9, 16)
		self.fc4 = nn.Linear(16, 9)
		self.fc5 = nn.Linear(9, 4)
		self.fc6 = nn.Linear(4, 1)

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x), 0.1)
		x = F.leaky_relu(self.fc2(x), 0.1)
		x = F.leaky_relu(self.fc3(x), 0.1)
		x = F.leaky_relu(self.fc4(x), 0.1)
		x = F.leaky_relu(self.fc5(x), 0.1)
		x = self.fc6(x)
		return x

def func(arr):
	return np.sin(5*np.pi*arr)/(5*np.pi*arr)
	# return np.exp(np.sin(40*arr))*np.log(arr+1)
	# return np.sin(3*np.pi*arr)+np.sin(4*np.pi*arr)

TRAIN_SIZE = 10000
BATCH_SIZE = 200
EPOCHS = 10


model = Net()

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
		loss = criterion(output, y)

		g = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
		g_sq = sum([grad.norm() ** 2 for grad in g])
		norm = g_sq.cpu().data.sqrt()

		loss.backward()
		optimizer.step()

		
		if (i+1) % 50 == 0:
			print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.6f} |\t grad_norm: {:.5f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))
			
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


optimizer = optim.Adam(model.parameters(), lr=0.00075)

for epoch in range(EPOCHS):
	for i, (x, y) in enumerate(train_loader):
		output = model(x)
		loss = criterion(output, y)
		g = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
		g_sq = sum([grad.norm() ** 2 for grad in g])
		optimizer.zero_grad()
		g_sq.backward(retain_graph=True)
		optimizer.step()


		if (i+1) % 50 == 0:
			norm = g_sq.cpu().data.sqrt()
			print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.6f} |\t grad_norm: {:.5f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))
			
	loss_hist.append(loss)
	norm_hist.append(norm)

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
n = torch.distributions.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
print('=============MODEL PARAM=============')
print(list(model.parameters())[0])


SAMPLE = 1
min_count = 0
eq_count = 0
for i in tqdm(range(SAMPLE)):
	model_cp = model
	# add random small num to params
	for param in model_cp.parameters():
		print('BEFORE:')
		print(param)
		rand = torch.ones_like(param)#n.sample(param.shape).squeeze()
		param = param + rand
		print('AFTER:')
		print(param)

###
	print(model.parameters() == model_cp.parameters())
	print('=============MODEL PARAM=============')
	print(model.parameters()[0])
	print('=============MODEL_CP PARAM=============')
	print(list(model.parameters())[0])

###
	m_loss = test(model_cp)
	if m_loss > test_loss: 
		min_count = min_count + 1
	elif m_loss == test_loss:
		eq_count = eq_count + 1


print('(min ratio, loss, eq_count): ({},{},{})'.format(min_count/SAMPLE, test_loss, eq_count))





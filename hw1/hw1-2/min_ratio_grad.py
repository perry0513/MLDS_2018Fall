import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from matplotlib import pyplot as plt

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

TRAIN_SIZE = 10000
BATCH_SIZE = 200
EPOCHS = 1000

def func(arr):
	return np.sin(5*np.pi*arr)/(5*np.pi*arr)
    #return np.exp(np.sin(40*arr))*np.log(arr+1)
	#return np.sin(3*np.pi*arr)+np.sin(4*np.pi*arr)

model = Net()
print(len(list(model.parameters())))

# pack data
x_train, x_test = torch.rand(TRAIN_SIZE, 1), torch.rand(TRAIN_SIZE, 1)
train_dataset = utils.TensorDataset(x_train, torch.from_numpy(func(x_train.detach().numpy())))
test_dataset  = utils.TensorDataset(x_test, torch.from_numpy(func(x_test.detach().numpy())))
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
			print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.4f} |\t grad_norm: {:.5f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))
			
	loss_hist.append(loss)
	norm_hist.append(norm)


with torch.no_grad():
	total_loss = 0
	for x, y in test_loader:
		output = model(x)
		loss = criterion(output, y)
		total_loss = total_loss + loss.numpy() / total_step
	print('Test loss: ', total_loss)


# plot
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

plt.subplot(2,1,2)
plt.plot(norm_hist)
plt.xlabel('epoch')
plt.ylabel('grad')

plt.show()


'''
# optim = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
			print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.4f} |\t grad_norm: {:.5f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item(), norm))
			
	loss_hist.append(loss)
	norm_hist.append(norm)








# plot
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

plt.subplot(2,1,2)
plt.plot(norm_hist)
plt.xlabel('epoch')
plt.ylabel('grad')

plt.show()

# test
with torch.no_grad():
	total_loss = 0
	for x, y in test_loader:
		output = model(x)
		loss = criterion(output, y)
		total_loss = total_loss + loss.numpy() / total_step
	print('Test loss: ', total_loss)
'''



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision

from mpl_toolkits.mplot3d import Axes3D

# Config
BATCH_SIZE = 600
EPOCHS = 50
NUM_COLOR = 1

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 32)
		self.fc5 = nn.Linear(32, 16)
		self.fc6 = nn.Linear(16, 10)


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc6(x)
		return x

model = Net()
if torch.cuda.is_available():
	model.cuda()
print(model)

# pack data
train_data = dset.MNIST(root='./MNIST-data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = dset.MNIST(root='./MNIST-data/', train=False, transform=torchvision.transforms.ToTensor())
train_loader = utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = utils.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

if torch.cuda.is_available:
	criterion.cuda()
# functions
def acc():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			images, labels = data
			if torch.cuda.is_available:
				images, labels = images.cuda(), labels.cuda()
			outputs = model(images.view(-1,28*28))
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		return correct / total

def weights_init(m):
	torch.manual_seed(np.random.randint(0,100))
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight.data)

# train
acc_hist_list = []
C_list = []
C_1_list = []

for t in range(NUM_COLOR):
	model.apply(weights_init)
	loss = 0
	loss_hist = []
	total_step = len(train_loader)
	params_hist = torch.Tensor()
	params_1_hist = torch.Tensor()
	acc_hist = []
	if torch.cuda.is_available:
		params_hist = params_hist.cuda()
		params_1_hist = params_1_hist.cuda()
	for epoch in range(EPOCHS):
		# collect params
		if epoch % 3 == 0:
			params = torch.Tensor()
			if torch.cuda.is_available:
				params = params.cuda()
			for p in model.parameters():
				params = torch.cat((params,p.view(1,-1)),1)
			params_1 = torch.cat((list(model.parameters())[0].view(1,-1),list(model.parameters())[1].view(1,-1)),1)
			params_hist = torch.cat((params_hist,params),0)
			params_1_hist = torch.cat((params_1_hist,params_1),0)
			acc_hist.append(acc()*100.)
			print ('accuracy: ', acc_hist[-1])
		for i, (x, y) in enumerate(train_loader):
			if torch.cuda.is_available:
				x,y = x.cuda(), y.cuda()
			optimizer.zero_grad()
			output = model(x.view(-1,28*28))
			loss = criterion(output, y.long())
			loss.backward()
			optimizer.step()
			if (i+1) % 50 == 0:
				print ('Epoch [{}/{}] | Step [{}/{}] |\t\tLoss: {:.4f}'.format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
			loss_hist.append(loss)
			
	U,S,V = torch.svd(torch.transpose(params_hist,0,1))
	C = torch.mm(params_hist,U[:,:2])
	C_list.append(C.transpose(0,1))
	U,S,V = torch.svd(torch.transpose(params_1_hist,0,1))
	C = torch.mm(params_1_hist,U[:,:2])
	C_1_list.append(C.transpose(0,1))
	acc_hist_list.append(acc_hist)

# test
with torch.no_grad():
	total_loss = 0
	for x, y in test_loader:
		if torch.cuda.is_available:
			x,y = x.cuda(),y.cuda()    		
		output = model(x.view(-1,28*28))
		loss = criterion(output, y)
		total_loss = total_loss + loss.cpu().numpy() / total_step
	print('Test loss: ', total_loss)

# plot
with np.errstate(invalid='ignore', divide='ignore') and torch.no_grad():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	c='r'
	m='o'
	x = C_list[0][0].cpu().detach().numpy()
	y = C_1_list[0][0].cpu().detach().numpy()
	X,Y = np.meshgrid(x, y)
	Z = acc_hist_list[0]
	ax.scatter(X, Y, Z, c=c, marker=m)
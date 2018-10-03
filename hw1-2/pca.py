import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision

# Config
BATCH_SIZE = 600
EPOCHS = 100

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 10)


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = Net()
if torch.cuda.is_available():
	model.cuda()
print(model)

# pack data
train_data = dset.MNIST(root='./MNIST-data/', train=True, transform=torchvision.transforms.ToTensor(), download=False)
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

for t in range(8):
	model.apply(weights_init)
	loss = 0
	loss_hist = []
	total_step = len(train_loader)
	params_hist = torch.Tensor()
	acc_hist = []
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
	ax1 = fig.add_subplot(111)
	for t in range(8):
		x_plot = C_list[t][0].cpu().detach().numpy()
		y_plot = C_list[t][1].cpu().detach().numpy()
		ax1.scatter(x_plot,y_plot,s=1,color='C'+str(t))
		for i,(j,k) in enumerate(zip(x_plot,y_plot)):
			ax1.annotate("%.2f"%acc_hist_list[t][i],xy=(j,k),fontsize=8,color='C'+str(t))
	plt.show()
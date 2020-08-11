'''
@Description: 
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-24 21:53:59
@LastEditors: Vicro
@LastEditTime: 2020-07-25 16:12:44
https://blog.csdn.net/davidsmith8/article/details/105938118/?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-2&spm=1001.2101.3001.4242
'''


from __future__ import print_function, division
import shutil
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import torch.optim as optim

torch.manual_seed(1)
epochs = 10
batch_size = 4
num_workers = 4
use_gpu = torch.cuda.is_available()

data_transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
 
 
train_dataset = datasets.ImageFolder(root = './data/train/', transform = data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_dataset = datasets.ImageFolder(root = './data/val/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers)
 
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.maxpool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 53 * 53, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 2)
	def forward(self, x):
		x = self.maxpool(F.relu(self.conv1(x)))
		x = self.maxpool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 53 * 53)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
if use_gpu:
	net = Net().cuda()
else:
	net = Net()
print(net)
 
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
 
net.train()
for epoch in range(epochs):
	running_loss = 0.0
	train_correct = 0
	train_total = 0
	for i, data in enumerate(train_loader, 0):
		inputs, train_labels = data
		if use_gpu:
			inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
		else:
			inputs, labels = Variable(inputs), Variable(train_labels)
 
		optimizer.zero_grad()
		outputs = net(inputs)
		_, train_predicted = torch.max(outputs.data, 1)
 
		train_correct += (train_predicted == labels.data).sum()
		loss = cirterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		train_total += train_labels.size(0)
	print('train %d epoch loss: %.3f acc: %.3f '% (epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
 
	correct = 0
	test_loss = 0.0
	test_total = 0
	net.eval()
	for data in test_loader:
		images, labels = data
		if use_gpu:
			images, labels = Variable(images.cuda()), Variable(labels.cuda())
		else:
			images, labels = Variable(images), Variable(labels)
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		loss = cirterion(outputs, labels)
		test_loss += loss.item()
		test_total += labels.size(0)
		correct += (predicted == labels.data).sum()

	print('test %d epoch loss: %.3f acc: %.3f' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))

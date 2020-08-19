'''
@Description: Test Cifar-10 mofel using checkpoint
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-25 22:58:37
LastEditors: Vicro
LastEditTime: 2020-08-20 00:25:36
https://blog.csdn.net/AugustMe/article/details/93917551?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase
'''
import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time
torch.manual_seed(1)
all_starttime = time.time()
BATCH_SIZE = 1
n_epochs = 120
# checkpoint_path = "./checkpoint/"

train_path = "./cifar10_train"
test_path =  "./cifar10_test"
transform = transforms.Compose([transforms.CenterCrop(32), # Crop from the middle
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]


test_image = datasets.ImageFolder(root = test_path, transform = transform)

testdata_loader_image = torch.utils.data.DataLoader(dataset=test_image,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)
# 检查电脑GPU资源
use_gpu = torch.cuda.is_available()
print(use_gpu) # 查看用没用GPU，用了打印True，没用打印False

# 加载模型并设为预训练
model = models.vgg19(pretrained = True)
print(model) # 查看模型结构

for parma in model.parameters():
    parma.requires_grad = False # 不进行梯度更新

# 改变模型的全连接层，因为原模型是输出1000个类，本项目只需要输出2类
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 10))

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
    
if use_gpu:
    model = model.cuda()

# 定义代价函数——交叉熵
cost = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.classifier.parameters())

# 再次查看模型结构
# print(model)


# model.load_state_dict(torch.load("./checkpoint/model240.pkl")) # 120: 60.92 # 240: 60.84
# model.load_state_dict(torch.load("./checkpoint_Ebbinghaus/model20.pkl")) # 20: 60.18  40: 60.76
# model.load_state_dict(torch.load("./checkpoint_data_agumentation/model120.pkl")) # 120: 60.03 240: 59.12 
# model.load_state_dict(torch.load("./small_checkpoint/model1597854121.9655204.pkl")) # 6: 12.45
model.load_state_dict(torch.load("./small_checkpoint_Ebbinghaus/model1597869868.3941066.pkl")) # 1: 9.83

Average_loss = 0.0
Average_correct = 0.0
Allepoch_batch = 0
test_epoch = 1

for epoch in range(test_epoch):
    # print("Epoch{}/{}".format(epoch + 1, n_epochs))
    # print("-"*10)
    model.train = False
        
    inepoch_batch = 0

    for data in testdata_loader_image:
        Step_loss = 0.0
        Step_correct = 0.0
        
        step_starttime = time.time()
        
        inepoch_batch += 1
        Allepoch_batch += 1

        X, y = data

        if use_gpu:
            X, y = Variable(X.cuda()), Variable(y.cuda())
        else:
            X,y = Variable(X), Variable(y)
        
        optimizer.zero_grad()

        y_pred = model(X)
        _,pred = torch.max(y_pred.data, 1)
        loss = cost(y_pred, y)

        Step_loss += loss.item()
        Average_loss += Step_loss

        step_time = time.time() - step_starttime
        all_time = time.time() - all_starttime

        Step_correct = float(torch.sum(pred == y.data))
        Average_correct += Step_correct

        if inepoch_batch%1 == 0:
            print("Epoch{}/{} Batch: {}  Ave_Loss: {:.5f}  Ave_Acc: {:.2f}  Step_Loss: {:.5f}  Step_Acc: {:.2f}  Step_Time: {:.3f} s  All_Time: {:.0f} min {:.2f} s".format(epoch + 1,
                                                                        n_epochs,
                                                                        inepoch_batch, 
                                                                        Average_loss / (BATCH_SIZE * Allepoch_batch), 
                                                                        100 * Average_correct / (BATCH_SIZE * Allepoch_batch),
                                                                        Step_loss / BATCH_SIZE, 
                                                                        100 * Step_correct / BATCH_SIZE,
                                                                        step_time % 60,
                                                                        all_time // 60,
                                                                        all_time % 60))

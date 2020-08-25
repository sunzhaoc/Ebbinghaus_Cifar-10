'''
@Description: Add Log and Loss
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-25 22:58:37
LastEditors: Vicro
LastEditTime: 2020-08-24 13:17:10
https://blog.csdn.net/AugustMe/article/details/93917551?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase
'''

import logging
import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time

logger = logging.getLogger('VGG16')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

torch.manual_seed(1)
all_starttime = time.time()
BATCH_SIZE = 200
n_epochs = 1000

checkpoint_path = "Z:/STUDY/checkpoint/checkpoint_Ebbinghaus/04/origin"
# train_path = "Z:/STUDY/cifar10_train"
train_path = "Y:/2020/Ebbinghaus-Cifar10/cifar10_train"
transform = transforms.Compose([transforms.CenterCrop(32), # Crop from the middle
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]

train_image = datasets.ImageFolder(root = train_path, transform = transform)

traindata_loader_image = torch.utils.data.DataLoader(dataset=train_image,
                                                batch_size = BATCH_SIZE,
                                                shuffle = True)
# 检查电脑GPU资源
use_gpu = torch.cuda.is_available()
print(use_gpu) # 查看用没用GPU，用了打印True，没用打印False

# 加载模型并设为预训练
model = models.vgg16(pretrained = True)
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
LR = 0.001
optimizer = torch.optim.Adam(model.classifier.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.95)

# 再次查看模型结构
# print(model)

### 开始训练模型
# model.load_state_dict(torch.load("./checkpoint/model20.pkl"))
Average_loss = 0.0
Average_correct = 0.0
Allepoch_batch = 0
All_batchsize = 0
step = 0

for epoch in range(n_epochs):
    model.train = True
        
    inepoch_batch = 0

    for data in traindata_loader_image:
        Step_loss = 0.0
        Step_correct = 0.0
                
        inepoch_batch += 1
        Allepoch_batch += 1
        step += 1

        X, y = data
        if use_gpu:
            X, y = Variable(X.cuda()), Variable(y.cuda())
        else:
            X,y = Variable(X), Variable(y)
        
        optimizer.zero_grad()

        y_pred = model(X)
        _,pred = torch.max(y_pred.data, 1)
        loss = cost(y_pred, y)


        loss.backward()
        optimizer.step()
        scheduler.step()
        Step_loss += loss.item()
        Average_loss += (Step_loss * BATCH_SIZE)

        all_time = time.time() - all_starttime

        Step_correct = float(torch.sum(pred == y.data))
        Average_correct += Step_correct
        All_batchsize += BATCH_SIZE
        if inepoch_batch%1 == 0:
            logger.info("Epoch{}/{} Step: {}  Ave_Loss: {:.5f}  Ave_Acc: {:.2f}  Step_Loss: {:.5f}  Step_Acc: {:.2f} Train_Time: {:.0f} h {:.0f} m {:.2f} s".format(
                epoch + 1,
                n_epochs,
                step,
                Average_loss / All_batchsize,
                100 * Average_correct / All_batchsize,
                Step_loss,
                100 * Step_correct / BATCH_SIZE,
                all_time // 3600,
                (all_time // 60)-60*(all_time // 3600),
                all_time % 60))

            logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='vgg16.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    # format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    format='%(asctime)s - %(levelname)s: %(message)s'
                    #日志格式
                    )
            
    if (epoch+1)%10 == 0: 
        torch.save(model.state_dict(), (checkpoint_path+"model_batch"+str(epoch*50000)+".pkl"))
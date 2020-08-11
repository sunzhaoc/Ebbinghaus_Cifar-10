'''
@Description: 
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-25 22:58:37
@LastEditors: Vicro
@LastEditTime: 2020-07-31 13:59:10
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

# BATCH_SIZE = 100
n_epochs = 1
folder_file = os.listdir('./NEWDATA')

# 检查电脑GPU资源
use_gpu = torch.cuda.is_available()
print(use_gpu) # 查看用没用GPU，用了打印True，没用打印False

# 加载模型并设为预训练
model = models.vgg19(pretrained = True)
# print(model) # 查看模型结构

for parma in model.parameters():
    parma.requires_grad = False # 不进行梯度更新

# 改变模型的全连接层，因为原模型是输出1000个类，本项目只需要输出2类
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
    
if use_gpu:
    model = model.cuda()

# 定义代价函数——交叉熵
cost = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.classifier.parameters())

# print(model)

### 开始训练模型
for epoch in range(n_epochs):
    # 每个batchsize
    batch = 0
    running_loss = 0.0
    running_correct = 0
    all_starttime = time.time()
    
    for path in folder_file:
        step_starttime = time.time()
        batch += 1

        # 计算batchsize
        BATCH_SIZE = 0;
        batch_path = os.path.join('./NEWDATA/', str(path))  #./NEWDATA/0000/
        batch_file = os.listdir(batch_path) # cat, dog
        for each_file in batch_file:
            sub_path = os.path.join(batch_path, str(each_file))  #./NEWDATA/0000/cat
            sub_file = os.listdir(sub_path)
            BATCH_SIZE += len(sub_file)
        # print(BATCH_SIZE)

        # 创建队列
        transform = transforms.Compose([transforms.CenterCrop(224), # Crop from the middle
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]

        data_image = datasets.ImageFolder(root = batch_path, transform = transform)

        data_loader_image = torch.utils.data.DataLoader(dataset=data_image,
                                                        batch_size = BATCH_SIZE,
                                                        shuffle = True)

        print("Epoch{}/{}".format(epoch, n_epochs))
        print("-"*10)

        model.train = True

        for data in data_loader_image:
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
            
            running_loss += loss.item()

            step_time = time.time() - step_starttime
            all_time = time.time() - all_starttime
            running_correct += torch.sum(pred == y.data)
            if batch%1 == 0:
                print("Batch: {}\tTrain Loss:{:.4f}\tTrain Acc:{:.3f}\tStep Time: {:.3f} s\tAll Time: {:.0f} min {:.2f} s".format(batch, 
                                                                            # running_loss / BATCH_SIZE, 
                                                                            #  100 * running_correct / BATCH_SIZE))
                                                                            running_loss / (BATCH_SIZE * batch), 
                                                                            100 * running_correct / (BATCH_SIZE * batch),
                                                                            step_time % 60,
                                                                            all_time // 60,
                                                                            all_time % 60))
                
        # epoch_loss = running_loss/len(data_image[param])
        # epoch_correct = 100*running_correct/len(data_image[param])
        
        # print("{} Loss:{:.4f}, Correct:{:.4f}".format(param, epoch_loss, epoch_correct))

    # print("Training time is:{:.0f}m {:.0f}s".format(now_time//60, now_time%60))

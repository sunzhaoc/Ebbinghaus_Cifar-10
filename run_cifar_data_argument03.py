'''
Description: Change vgg classify
Version: 1.0
Autor: Vicro
Date: 2020-09-02 21:48:19
LastEditors: Vicro
LastEditTime: 2020-09-02 21:58:11
'''



import logging
import torch
import torch.nn as nn
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
n_epochs = 100
checkpoint_path = "Z:/STUDY/checkpoint/checkpoint_Ebbinghaus/05/data_argument/"
train_path = "Y:/2020/Ebbinghaus-Cifar10/cifar10_train"
WHETHER_MODEL = False
WHETHER_LOG = False
LOG_NAME = 'vgg16_pro_data_argument.txt'


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )

        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            #16
            nn.Linear(256,num_classes),
            )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]

train_image = datasets.ImageFolder(root = train_path, transform = transform)

traindata_loader_image = torch.utils.data.DataLoader(dataset=train_image,
                                                     batch_size = BATCH_SIZE,
                                                     shuffle = True,
                                                     num_workers = 0
                                                     )

model = models.vgg16(pretrained = True)


for parma in model.parameters():
    parma.requires_grad = True # 不进行梯度更新

model.avgpool = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=1, stride=1))
model.classifier = torch.nn.Sequential(torch.nn.Linear(512, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 10))
print(model)

# Check GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
    print(use_gpu)

# Define Loss and Optimizer
cost = torch.nn.CrossEntropyLoss()
LR = 0.001
optimizer = torch.optim.Adam(model.classifier.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.95)

### 开始训练模型
# model.load_state_dict(torch.load("./checkpoint/model20.pkl"))
Average_loss = 0.0
Average_correct = 0.0
All_step = 0
All_input_pic = 0


def Train():
    global Average_loss
    global All_step
    global Average_correct
    global All_input_pic

    for epoch in range(n_epochs):

        model.train = True
        inepoch_step = 0

        for data in traindata_loader_image:
            Step_loss = 0.0
            Step_correct = 0.0

            inepoch_step += 1
            All_step += 1

            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X,y = Variable(X), Variable(y)

            y_pred = model(X)
            _,pred = torch.max(y_pred.data, 1)
            loss = cost(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            Step_loss += loss.item()
            Average_loss += (Step_loss * BATCH_SIZE)

            all_time = time.time() - all_starttime

            Step_correct = float(torch.sum(pred == y.data))
            Average_correct += Step_correct
            All_input_pic += BATCH_SIZE
            if inepoch_step%1 == 0:
                logger.info("Epoch{}/{} Step: {}  Ave_Loss: {:.5f}  Ave_Acc: {:.2f}  Step_Loss: {:.5f}  Step_Acc: {:.2f} Train_Time: {:.0f} h {:.0f} m {:.2f} s".format(
                    epoch + 1,
                    n_epochs,
                    All_step,
                    Average_loss / All_input_pic,
                    100 * Average_correct / All_input_pic,
                    Step_loss,
                    100 * Step_correct / BATCH_SIZE,
                    all_time // 3600,
                    (all_time // 60) - 60 * (all_time // 3600),
                    all_time % 60))

                if WHETHER_LOG:
                    logging.basicConfig(level=logging.INFO,
                                        filename=LOG_NAME,
                                        filemode='a',
                                        format='%(asctime)s - %(levelname)s: %(message)s')

        if (epoch+1)%20 == 0:
            if WHETHER_MODEL:
                torch.save(model.state_dict(), (checkpoint_path+"model_epoch"+str(epoch+1)+".pkl"))


if __name__ == "__main__":
    Train()
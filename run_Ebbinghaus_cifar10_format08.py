'''
Description: 
Version: 1.0
Autor: Vicro
Date: 2020-08-20 16:54:45
LastEditors: Vicro
LastEditTime: 2020-08-20 20:58:33
'''


# Import package
import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time

torch.manual_seed(1)    # Set random seed

n_epochs = 1
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
SMALL_BATCH_SIZE = 5
all_starttime = time.time()


""" Construct Queue Dict """
for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
    for num in [0, 1, 2, 4, 7, 15]:
        if (i==0) and (num==0):
            # queue_dict[str(i+num)] = all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]
            queue_dict = {str(i): all_data_file[i*SMALL_BATCH_SIZE: i*SMALL_BATCH_SIZE+SMALL_BATCH_SIZE]}
        else:
            if str(i+num) in queue_dict.keys():
               for j in range(SMALL_BATCH_SIZE):
                    # queue_dict[str(i+num)].append(all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE])
                    queue_dict[str(i+num)].append(all_data_file[i*SMALL_BATCH_SIZE+j])
            else:
                queue_dict[str(i+num)] = all_data_file[i*SMALL_BATCH_SIZE: i*SMALL_BATCH_SIZE+SMALL_BATCH_SIZE]

use_gpu = torch.cuda.is_available() # Check GPU
print(use_gpu)

# Load Model
model = models.vgg19(pretrained = True)
# print(model) 
for parma in model.parameters():
    parma.requires_grad = False # 不进行梯度更新
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
cost = torch.nn.CrossEntropyLoss()  # 定义代价函数——交叉熵
optimizer = torch.optim.Adam(model.classifier.parameters())
# print(model)

Average_loss = 0.0
Average_correct = 0
Allepoch_batch = 0
All_batchsize = 0

# def train_Ebbinghaus(NOW_BATCH):
    
#     global Average_loss
#     global Average_correct
#     global Allepoch_batch
#     global All_batchsize
for xunhuancishu in range(10):
    for NOW_BATCH in range(len(all_data_file)//SMALL_BATCH_SIZE):
        # 1. Delete Data
        for i in os.listdir('Z:/STUDY/tempcifar10'):
            for j in os.listdir(os.path.join('Z:/STUDY/tempcifar10', i)):
                del_path = os.path.join('Z:/STUDY/tempcifar10', i, j)
                os.remove(del_path)

        # 2. Move Data
        try:
            for i in queue_dict[str(NOW_BATCH)]:
                if "airplane" in i: # 1
                    tag = "airplane"
                elif "automobile" in i: # 2
                    tag = "automobile"
                elif "bird" in i:   # 3
                    tag = "bird"
                elif "cat" in i:    # 4
                    tag = "cat"
                elif "deer" in i:   # 5
                    tag = "deer"
                elif "dog" in i:    # 6
                    tag = "dog"
                elif "frog" in i:   # 7
                    tag = "frog"
                elif "horse" in i:  # 8
                    tag = "horse"
                elif "ship" in i:   # 9
                    tag = "ship"
                elif "truck" in i:  # 10
                    tag = "truck"

                old_dir = os.path.join("Z:/STUDY/all_cifar/", i)
                # print(old_dir)
                new_dir = os.path.join("Z:/STUDY/tempcifar10/", tag)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                new_dir = os.path.join("Z:/STUDY/tempcifar10/", tag, i)
                shutil.copyfile(old_dir, new_dir)
        except KeyError:
            print("-----Save model-----")
            torch.save(model.state_dict(), ("Z:/STUDY/checkpoint_Ebbinghaus_format02/model"+str(xunhuancishu)+".pkl"))
            break
        # 3. Caculate BATCHSIZE
        BATCH_SIZE = 0;
        for each_file in os.listdir(os.path.join('Z:/STUDY/tempcifar10')):
            sub_path = os.path.join(os.path.join('Z:/STUDY/tempcifar10'), str(each_file))  #./NEWDATA/0000/cat
            sub_file = os.listdir(sub_path)
            BATCH_SIZE += len(sub_file)
        # print(BATCH_SIZE)
        All_batchsize += BATCH_SIZE

        # 4. Load Data
        transform = transforms.Compose([transforms.CenterCrop(32), # Crop from the middle
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]
        
        try:
            data_image = datasets.ImageFolder(root = 'Z:/STUDY/tempcifar10', transform = transform)
        except RuntimeError:
            continue
        data_loader_image = torch.utils.data.DataLoader(dataset=data_image,
                                                        batch_size = BATCH_SIZE,
                                                        shuffle = True)
        
        # 5. Train
        model.train = True
        step_starttime = time.time()
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

            Step_loss = loss.item()
            Average_loss += Step_loss

            step_time = time.time() - step_starttime
            all_time = time.time() - all_starttime

            Step_correct = float(torch.sum(pred == y.data))
            Average_correct += Step_correct
            
            print("Epoch: {}  Batch: {}  BATCHSIZE: {} Ave_Loss: {:.5f}  Ave_Acc: {:.2f}  Step_Loss: {:.5f}  Step_Acc: {:.2f}  Step_Time: {:.3f} s  All_Time: {:.0f} min {:.2f} s  AllBATCH: {}".format(
                                                                        xunhuancishu+1,
                                                                        NOW_BATCH, 
                                                                        BATCH_SIZE,
                                                                        Average_loss / All_batchsize, 
                                                                        100 * Average_correct / All_batchsize,
                                                                        Step_loss / BATCH_SIZE, 
                                                                        100 * Step_correct / BATCH_SIZE,
                                                                        step_time % 60,
                                                                        all_time // 60,
                                                                        all_time % 60,
                                                                        All_batchsize))
        
        # 6. Adjust Data
        y = y.tolist()
        y_predict = pred.tolist()
        
        same_index = []
        dif_index = []

        for i in range(len(y_predict)):
            if y[i] == y_predict[i]:
                same_index.append(i)
            else:
                dif_index.append(i)


        """ Update Queue Dict"""
        for i in dif_index:
            # Delete Element
            for num in range(15):
                if str(NOW_BATCH+num+1) in queue_dict.keys():
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        queue_dict[str(NOW_BATCH+num+1)].remove(queue_dict[str(NOW_BATCH)][i])
                else:
                    queue_dict[str(NOW_BATCH+num+1)] = []
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        queue_dict[str(NOW_BATCH+num+1)].remove(queue_dict[str(NOW_BATCH)][i])
            # Append Element
            for num in [1, 2, 4, 7, 15]:
                if str(NOW_BATCH+num) in queue_dict.keys():
                    queue_dict[str(NOW_BATCH+num)].append(queue_dict[str(NOW_BATCH)][i])
                else:
                    queue_dict[str(NOW_BATCH+num)] = [queue_dict[str(NOW_BATCH)][i]]
        
        del_index = []
        for i in same_index:
            for num in range(15):
                if str(NOW_BATCH+num+1) in queue_dict.keys():
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        del_index.append(i)
                        break

        del_index.extend(dif_index)
        del_index.sort(reverse = True)

        for i in del_index:
            del queue_dict[str(NOW_BATCH)][i]

'''
Description: Dlete Same Data in Queue
Version: 1.0
Autor: Vicro
Date: 2020-08-20 22:42:20
LastEditors: Vicro
LastEditTime: 2020-08-21 23:20:59
'''


# Import package
import logging
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

# Logging information
logger = logging.getLogger('Ebbinghaus')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

torch.manual_seed(1)    # Set random seed

n_epochs = 1    # train epoch
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
SMALL_BATCH_SIZE = 200    
MAX_BATCH_SIZE = 200
all_starttime = time.time()


""" Construct Queue Dict """
for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
# for i in range(20):
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
model = models.vgg16(pretrained = True)
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

LR = 0.001
optimizer = torch.optim.Adam(model.classifier.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.95)
lr_list = []

# print(model)

Average_loss = 0.0
Average_correct = 0
All_step = 0
All_input_pic = 0
All_batch = 0
for xunhuancishu in range(4):
    for NOW_BATCH in range(99999999):
        # 1. Delete Data
        for i in os.listdir('Z:/STUDY/tempcifar10'):
            for j in os.listdir(os.path.join('Z:/STUDY/tempcifar10', i)):
                del_path = os.path.join('Z:/STUDY/tempcifar10', i, j)
                os.remove(del_path)

        
        try:
        # 1.5 Adjust Queue
            if len(queue_dict[str(NOW_BATCH)]) < MAX_BATCH_SIZE:
                for i in range(len(queue_dict)):
                    for j in range(len(queue_dict[str(i+1)])):
                        queue_dict[str(NOW_BATCH)] += [queue_dict[str(i+1)][0]]
                        del queue_dict[str(i+1)][0]

                        temp_dict = []
                        [temp_dict.append(k) for k in queue_dict[str(NOW_BATCH)] if not k in temp_dict]
                        queue_dict[str(NOW_BATCH)] = temp_dict

                        if len(queue_dict[str(NOW_BATCH)]) >= MAX_BATCH_SIZE:
                            break
                    if len(queue_dict[str(NOW_BATCH)]) >= MAX_BATCH_SIZE:
                        break
                
                # Caculate Move Step
                move_step = 0
                for i in range(len(queue_dict)):
                    if len(queue_dict[str(NOW_BATCH + i + 1)]) == 0:
                        move_step += 1
                    else:
                        break
                
                # Adjust Queue
                for i in range(NOW_BATCH+1, len(queue_dict)-move_step):
                    queue_dict[str(i)] = queue_dict[str(i+move_step)]
                for i in range(len(queue_dict)-move_step, len(queue_dict)):
                    del queue_dict[str(i)]
            
            if len(queue_dict[str(NOW_BATCH)]) > MAX_BATCH_SIZE:
                del_num = len(queue_dict[str(NOW_BATCH)]) - MAX_BATCH_SIZE
                temp_dict = queue_dict[str(NOW_BATCH)][-del_num:]
                temp_dict += queue_dict[str(NOW_BATCH+1)]
                queue_dict[str(NOW_BATCH+1)] = temp_dict
                # Delete Same Data
                temp_dict = []
                [temp_dict.append(i) for i in queue_dict[str(NOW_BATCH+1)] if not i in temp_dict]
                queue_dict[str(NOW_BATCH+1)] = temp_dict
                del queue_dict[str(NOW_BATCH)][-del_num:]

        # 2. Move Data
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
                new_dir = "Z:/STUDY/tempcifar10/" +  tag + '/' + i[:-4] + str(time.time()) + ".png"
                shutil.copyfile(old_dir, new_dir)
        except KeyError:
            print("-----Save model-----")
            torch.save(model.state_dict(), ("Z:/STUDY/checkpoint/model_epoch"+str(xunhuancishu)+".pkl"))
            break

        # 3. Caculate BATCHSIZE
        BATCH_SIZE = 0;
        for each_file in os.listdir(os.path.join('Z:/STUDY/tempcifar10')):
            sub_path = os.path.join(os.path.join('Z:/STUDY/tempcifar10'), str(each_file))  #./NEWDATA/0000/cat
            sub_file = os.listdir(sub_path)
            BATCH_SIZE += len(sub_file)
        # print(BATCH_SIZE)
        All_input_pic += BATCH_SIZE
        # BATCH_SIZE = MAX_BATCH_SIZE

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
            
            scheduler.step()

            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

            Step_loss = loss.item()
            Average_loss += (Step_loss * BATCH_SIZE)

            step_time = time.time() - step_starttime
            all_time = time.time() - all_starttime
            All_batch += 1
            Step_correct = float(torch.sum(pred == y.data))
            Average_correct += Step_correct
        
            logger.info("Epoch: {} Batch: {} Batchsize: {} Ave_Loss: {:.5f} Ave_Acc: {:.2f} Step_Loss: {:.5f} Step_Acc: {:.2f} Step_Time: {:.3f}s All_Time: {:.0f} min {:.2f} s All_data: {} All_batch: {} Q_Length: {} {}".format(
                xunhuancishu + 1,
                                                                        NOW_BATCH, 
                                                                        BATCH_SIZE,
                Average_loss / All_input_pic,
                100 * Average_correct / All_input_pic,
                                                                        Step_loss,
                100 * Step_correct / BATCH_SIZE,
                step_time % 60,
                all_time // 60,
                all_time % 60,
                                                                        All_input_pic,
                                                                        All_batch,
                                                                        len(queue_dict),
                                                                        len(queue_dict[str(NOW_BATCH+1)])
                                                                        ))

            # logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
            #         filename='new_ebbinghaus_newbatch_oldqueue.log',
            #         filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
            #         #a是追加模式，默认如果不写的话，就是追加模式
            #         # format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
            #         format='%(asctime)s - %(levelname)s: %(message)s'
            #         #日志格式
            #         )
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
        if xunhuancishu==0:
            liebiao = [1, 2, 4, 7, 15]
            max_liebiao = liebiao[4]
        
        for i in dif_index:
            # Delete Element
            for num in range(max_liebiao):
                if str(NOW_BATCH+num+1) in queue_dict.keys():
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        queue_dict[str(NOW_BATCH+num+1)].remove(queue_dict[str(NOW_BATCH)][i])
                else:
                    queue_dict[str(NOW_BATCH+num+1)] = []
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        queue_dict[str(NOW_BATCH+num+1)].remove(queue_dict[str(NOW_BATCH)][i])
            # Append Element
            for num in liebiao:
                if str(NOW_BATCH+num) in queue_dict.keys():
                    queue_dict[str(NOW_BATCH+num)].append(queue_dict[str(NOW_BATCH)][i])
                else:
                    queue_dict[str(NOW_BATCH+num)] = [queue_dict[str(NOW_BATCH)][i]]
        
        del_index = []
        for i in same_index:
            for num in range(max_liebiao):
                if str(NOW_BATCH+num+1) in queue_dict.keys():
                    if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
                        del_index.append(i)
                        break

        del_index.extend(dif_index)
        del_index.sort(reverse = True)

        for i in del_index:
            del queue_dict[str(NOW_BATCH)][i]
        
        # if (All_batchsize%50000==0):
        #     torch.save(model.state_dict(), ("Z:/STUDY/checkpoint/model_batch"+str(All_batchsize)+".pkl"))

    
    time.sleep(90)

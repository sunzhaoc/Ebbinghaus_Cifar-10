'''
Description:  A2不走，失败
Version: 1.0
Autor: Vicro
Date: 2020-08-22 19:49:19
LastEditors: Vicro
LastEditTime: 2020-08-25 15:40:08
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


WHETHER_LOG = False
WHETHER_SAVE_MODEL = True

ORIGIN_BATCHSIZE = 500
BATCH_SIZE = 200

ebbinghaus_list = [0, 1, 6, 144, 288, 576]

add_liebiao = ebbinghaus_list[1:]
del_liebiao = [0]
for i in range(len(ebbinghaus_list)):
    for j in range(i+1, len(ebbinghaus_list)):
        del_liebiao.append(ebbinghaus_list[j]-ebbinghaus_list[i])
        del_liebiao.append(ebbinghaus_list[j]-ebbinghaus_list[i]-1) # 针对写定batch大小
del_liebiao = list(set(del_liebiao))

def Construct_Queue_Dict(data, batch_size):
    queue_dict = {}
    for i in range(len(data)//batch_size):
        temp_data = data[i*batch_size: i*batch_size+batch_size]
        temp_ebbinghaus_list = [k+i for k in ebbinghaus_list]
        for j in temp_data:
            queue_dict[str(j)] = temp_ebbinghaus_list
    return queue_dict


def Construct_Now_List(data, step):
    temp_list = []
    for i in data:
        if step in data[i]:
            temp_list.append(i)
    return temp_list
    

def Delete_Move_Data(list):
    # 1. Delete Data
    for i in os.listdir('Z:/STUDY/tempcifar10'):
        for j in os.listdir(os.path.join('Z:/STUDY/tempcifar10', i)):
            del_path = os.path.join('Z:/STUDY/tempcifar10', i, j)
            os.remove(del_path)
    # 2. Move Data
    for i in list:
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


def Delete_Move_Data2(list, step):
    for i in os.listdir('Z:/STUDY/tempcifar10'):
        for j in os.listdir(os.path.join('Z:/STUDY/tempcifar10', i)):
            del_path = os.path.join('Z:/STUDY/tempcifar10', i, j)
            os.remove(del_path)
    try:
        for i in list[str(step)]:
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
        pass


def Updata_Queue_Dict(data, now_list, y, y_predict, step):
    same_index = []
    dif_index = []
    for i in range(len(y_predict)):
        same_index.append(i) if y[i] == y_predict[i] else dif_index.append(i)

    dif_data = []
    same_data = []
    for i in dif_index:
        dif_data.append(now_list[i])
    for i in same_index:
        same_data.append(now_list[i])

    dif_ebbinghaus_list = [k+step+1 for k in ebbinghaus_list]
    # False Data
    for i in dif_data:
        data[i] = dif_ebbinghaus_list
    
    # True Data
    for i in same_data:
        data[i] = data[i][data[i].index(step)+1:]
        # Make Sure Every Data will not be deleted completed
        if not data[i]:
            data[i] = [step+4320]
        
    return data


def Updata_Queue_Dict2(queue_dict, y, y_predict, NOW_BATCH):
    same_index = []
    dif_index = []

    for i in range(len(y_predict)):
        same_index.append(i) if y[i] == y_predict[i] else dif_index.append(i)
        # if y[i] == y_predict[i]:
        #     same_index.append(i)
        # else:
        #     dif_index.append(i)


    """ Update Queue Dict"""
    # For Wrong Data


    dif_data = []
    for i in dif_index:
        dif_data.append(queue_dict[str(NOW_BATCH)][i])
    dif_data.reverse()

    for i in dif_data:
        # Delete Element
        for num in del_liebiao:
            try:
                try:
                    queue_dict[str(NOW_BATCH+num)].remove(i)
                except ValueError:
                    continue
            except KeyError:
                continue
        
        # Append Element
        for num in add_liebiao:
            try:
                queue_dict[str(NOW_BATCH+num)].insert(0, i)
            except KeyError:
                queue_dict[str(NOW_BATCH+num)] = [i]
                continue

    # For True Data
    same_data = queue_dict[str(NOW_BATCH)]

    del_num = []
    for i in same_data:
        for num in del_liebiao[1:]:
            try:
                if i in queue_dict[str(NOW_BATCH+num)]:
                    # queue_dict[str(NOW_BATCH)].remove(i)
                    del_num.append(i)
                    break
            except KeyError:
                continue
    for i in del_num:
        queue_dict[str(NOW_BATCH)].remove(i)

    return queue_dict


def Push_Batchsize(data, step, batch_size):
    global Push_to_pull
    now_list = Construct_Now_List(data, step)

    if not Push_to_pull:
        if len(now_list)>batch_size:
            print("Push")
            more_list = now_list[batch_size-len(now_list):]
            for i in more_list:
                more_ebbinghaus_list = [j+1 for j in data[i]]
                data[i] = more_ebbinghaus_list
        else:
            Push_to_pull = True

            max_step = []
            min_step = []
            for i in data:
                max_step.append(max(data[i]))
                min_step.append(min(data[i]))
            max_step = max(max_step)
            min_step = min(min_step)
            
            data2 = {}
            for i in range(min_step, max_step+1):
                now_list = Construct_Now_List(data, i)
                data2[str(i)] = now_list
                print(i)
            data = data2
    return data


def Pull_Batchsize(data, step, batch_size):
    if len(data[str(step)]) < batch_size:
        for i in range(step+1, len(data)):
            for j in range(len(data[str(i)])):
                data[str(step)] += [data[str(i)][0]]
                del data[str(i)][0]

                temp_dict = []
                [temp_dict.append(k) for k in data[str(step)] if not k in temp_dict]
                data[str(step)] = temp_dict

                if len(data[str(step)]) >= batch_size:
                    break
            if len(data[str(step)]) >= batch_size:
                break
        
        # Caculate Move Step
        move_step = 0
        for i in range(step+1, len(data)):
            if len(data[str(i)]) == 0:
                move_step += 1
            else:
                break
        
        # Adjust Queue
        for i in range(step+1, len(data)-move_step):
            data[str(i)] = data[str(i+move_step)]
        for i in range(len(data)-move_step, len(data)):
            del data[str(i)]
        return data
    else:
        return data


def Model_Set():
    # Load Model
    model = models.vgg16(pretrained = True)

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
    use_gpu = torch.cuda.is_available() # Check GPU
    if use_gpu:
        model = model.cuda()
        print("Use GPU")
    cost = torch.nn.CrossEntropyLoss()  # 定义代价函数——交叉熵

    LR = 0.001
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.95)

    return model, optimizer, scheduler, cost, use_gpu


def Train(step, model, BATCH_SIZE, optimizer, scheduler, cost, use_gpu):
    global logger

    # Load Data
    transform = transforms.Compose([transforms.CenterCrop(32), # Crop from the middle
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]) # Let Tensor from [0, 1] to [-1, 1]
    data_image = datasets.ImageFolder(root = 'Z:/STUDY/tempcifar10', transform = transform)
    data_loader_image = torch.utils.data.DataLoader(dataset=data_image,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
    model.train = True
    global Average_loss
    global Average_correct                                
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

        Step_loss = loss.item()
        Average_loss += (Step_loss * BATCH_SIZE)

        Step_correct = float(torch.sum(pred == y.data))
        Average_correct += Step_correct

        logger.info("Step: {} Batchsize: {} Ave_Loss: {:.5f} Ave_Acc: {:.2f} % Step_Loss: {:.5f} Step_Acc: {:.2f} % All_data: {} Train_Time: {:.0f} h {:.0f} m {:.2f} s".format(
            step,
            BATCH_SIZE,
            Average_loss / ((step+1)*BATCH_SIZE),
            100 * Average_correct / ((step+1)*BATCH_SIZE),
            Step_loss,
            100 * Step_correct / BATCH_SIZE,
            (step+1)*BATCH_SIZE,
            (time.time()-start_time) // 3600,   # hour
            ((time.time()-start_time) // 60) - 60*((time.time()-start_time) // 3600),   # min
            (time.time()-start_time) % 60,    # sec
            ))
        if WHETHER_LOG: 
            logging.basicConfig(level=logging.INFO,
                                filename='ebbinghaus.log',
                                filemode='a',
                                format='%(asctime)s - %(levelname)s: %(message)s')

        y = y.tolist()
        y_predict = pred.tolist()
        

        return y, y_predict, (step+1)*BATCH_SIZE


def Data_Usage_Rate(data):
    global Used_data
    # Used data
    if len(Used_data)<50000:
        Used_data += data
        Used_data = list(set(Used_data))
        return (100*len(Used_data)/50000)
    else:
        return (100)


torch.manual_seed(1)    # Set random seed
logger = logging.getLogger("Ebbinghaus")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory

Average_loss = 0.0
Average_correct = 0.0
start_time = time.time()
Used_data = []
Push_to_pull = False

model, optimizer, scheduler, cost, use_gpu = Model_Set()
queue_dict = Construct_Queue_Dict(all_data_file, ORIGIN_BATCHSIZE)
for step in range(99999999):
    if not Push_to_pull:
        queue_dict = Push_Batchsize(queue_dict, step, BATCH_SIZE)
    else:
        queue_dict = Pull_Batchsize(queue_dict, step, BATCH_SIZE)
    
    if not Push_to_pull:
        now_list = Construct_Now_List(queue_dict, step)
        Delete_Move_Data(now_list)
    else:
        Delete_Move_Data2(queue_dict, step)
    
    if not Push_to_pull:
        y, y_predict, used_data = Train(step, model, BATCH_SIZE, optimizer, scheduler, cost, use_gpu)
        queue_dict = Updata_Queue_Dict(queue_dict, now_list, y, y_predict, step)
    else:    
        y, y_predict, used_data = Train(step, model, BATCH_SIZE, optimizer, scheduler, cost, use_gpu)
        queue_dict = Updata_Queue_Dict2(queue_dict, y, y_predict, step)

    

    # Data_Usage_Rate(now_list)
    # if used_data%500000==0:
    #     if WHETHER_SAVE_MODEL:
    #         torch.save(model.state_dict(), ("Z:/STUDY/checkpoint/model_batch" + str((step+1)*BATCH_SIZE) + ".pkl"))              
'''
Description:  Ebbinghaus Version 2.0
Version: 2.0
Autor: Vicro
Date: 2020-08-20 06:30:28
LastEditors: Vicro
LastEditTime: 2020-08-20 19:50:42
'''

import os
import shutil
import random
import torch
import torchvision
import operator
from torchvision import datasets, transforms, models


all_data_file = os.listdir('./all_cifar/')
BATCH_SIZE = 1
NOW_BATCH = 4

""" Construct Queue Dict """
# for i in range(len(all_data_file)//BATCH_SIZE):
#     if i == 0:
#         queue_dict = {str(i): all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]}
#     else:
#         queue_dict[str(i)] = all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]

##### TEST
for i in range(100):
    for num in [0, 1, 2, 4, 7, 15]:
        if (i==0) and (num==0):
            queue_dict = {str(i): all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]}
        else:
            if str(i+num) in queue_dict.keys():
                for j in range(BATCH_SIZE):
                    queue_dict[str(i+num)].append(all_data_file[i*BATCH_SIZE+j])
            else:
                queue_dict[str(i+num)] = all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]




# for i in os.listdir('./tempcifar10'):
#     for j in os.listdir(os.path.join('./tempcifar10', i)):
#         del_path = os.path.join('./tempcifar10', i, j)
#         os.remove(del_path)




# for i in queue_dict["0"]:
#     if "airplane" in i: # 1
#         tag = "airplane"
#     elif "automobile" in i: # 2
#         tag = "automobile"
#     elif "bird" in i:   # 3
#         tag = "bird"
#     elif "cat" in i:    # 4
#         tag = "cat"
#     elif "deer" in i:   # 5
#         tag = "deer"
#     elif "dog" in i:    # 6
#         tag = "dog"
#     elif "frog" in i:   # 7
#         tag = "frog"
#     elif "horse" in i:  # 8
#         tag = "horse"
#     elif "ship" in i:   # 9
#         tag = "ship"
#     elif "truck" in i:  # 10
#         tag = "truck"

#     old_dir = os.path.join("./all_cifar/", i)
#     # print(old_dir)
#     new_dir = os.path.join("./tempcifar10/", tag)
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)
#     new_dir = os.path.join("./tempcifar10/", tag, i)
#     shutil.copyfile(old_dir, new_dir)
    

y_predict = [4, 2, 3, 1]
y = [3, 2, 1, 4]


"""Construct Same and Different Index"""
same_index = []
dif_index = []

for i in range(len(y_predict)):
    if y[i] == y_predict[i]:
        same_index.append(i)
    else:
        dif_index.append(i)


""" Update Queue Dict"""
# for i in dif_index:
#     # Delete Element
#     for num in range(14):
#         if queue_dict[str(NOW_BATCH)][i] in queue_dict[str(NOW_BATCH+num+1)]:
#             queue_dict[str(NOW_BATCH+num+1)].remove(queue_dict[str(NOW_BATCH)][i])
#     # Append Element
#     for num in [1, 2, 4, 7, 15]:
#         queue_dict[str(NOW_BATCH+num)].append(queue_dict[str(NOW_BATCH)][i])

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
# print(list)

a=1

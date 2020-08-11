'''
@Description: 
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-26 16:49:55
@LastEditors: Vicro
@LastEditTime: 2020-07-31 19:14:30
'''
import os
import shutil
import random
import torch
import torchvision
from torchvision import datasets, transforms, models

# path = "./NEWDATA/"
# for name in range(0, 2265):
#     new_dir = os.path.join(path, (str(name+1)).zfill(4))
#     print(new_dir)
#     os.mkdir(new_dir)

# dog_file = os.listdir('./data/train/dog')
# cat_file = os.listdir('./data/train/cat')
# data_file = dog_file + cat_file
# random.shuffle(data_file)
# BATCH_SIZE = 10
# print(len(data_file))

BATCH_SIZE = 100
all_data_file = os.listdir('./all_cifar/')
random.shuffle(all_data_file)
# print(all_data_file)
data_file = os.listdir('./CIFAR10')
# print(len(datafile))


for batch in range(len(all_data_file) // BATCH_SIZE):
    print("{:.2f}%".format(batch * 100 / (len(all_data_file) // BATCH_SIZE)))
    # print(len(all_data_file) // BATCH_SIZE)
    batch_file = all_data_file[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
    # print(len(batch_file))
    for i in batch_file:

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

        old_dir = os.path.join("./CIFAR10", tag, i)
        # print(old_dir)
        for num in [0, 1, 2, 4, 7, 15]:
            new_dir = os.path.join("./new_cifar10/", (str(batch + num)).zfill(4), tag)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_dir = os.path.join(new_dir, i)
            shutil.copyfile(old_dir, new_dir)
            # print(new_dir)
            a=1
        # print(new_dir)
        a=1
    a=1

# 计算batchsize
# folder_file = os.listdir('./NEWDATA/')  # 0000, 0001, 0002
# for path in folder_file:
#     # 计算batchsize
#     BATCH_SIZE = 0;
#     batch_path = os.path.join('./NEWDATA/', str(path))  #./NEWDATA/0000/
#     batch_file = os.listdir(batch_path) # cat, dog
#     for each_file in batch_file:
#         sub_path = os.path.join(batch_path, str(each_file))  #./NEWDATA/0000/cat
#         sub_file = os.listdir(sub_path)
#         BATCH_SIZE += len(sub_file)
#     print(BATCH_SIZE)
#     a=1


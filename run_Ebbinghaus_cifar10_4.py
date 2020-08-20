'''
Description:  Ebbinghaus Version 2.0
Version: 2.0
Autor: Vicro
Date: 2020-08-20 06:30:28
LastEditors: Vicro
LastEditTime: 2020-08-20 06:38:01
'''

import os
import shutil
import random
import torch
import torchvision
from torchvision import datasets, transforms, models


all_data_file = os.listdir('./all_cifar/')

for i in range(5):
    data_dict = {str(i): all_data_file[i*10: i*10+5]}
print(data_dict)
a =1
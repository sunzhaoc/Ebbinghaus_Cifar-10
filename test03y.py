'''
Description: 
Version: 1.0
Autor: Vicro
Date: 2020-08-22 10:33:57
LastEditors: Vicro
LastEditTime: 2020-08-22 18:42:29
'''

import os
import time

starttime = time.time()
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
SMALL_BATCH_SIZE = 4
NOW_BATCH = 0


# for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
for i in range(20):
    for num in [0,1,6,144,288,576,1152,2016,4320]:
        if (i==0) and (num==0):
            # queue_dict[str(i+num)] = all_data_file[i*BATCH_SIZE: i*BATCH_SIZE+BATCH_SIZE]
            queue_dict = {str(i): all_data_file[i*SMALL_BATCH_SIZE: i*SMALL_BATCH_SIZE+SMALL_BATCH_SIZE]}
        else:
            if str(i+num) in queue_dict.keys():
                queue_dict[str(i+num)].extend(all_data_file[i*SMALL_BATCH_SIZE: i*SMALL_BATCH_SIZE+SMALL_BATCH_SIZE])
                # for j in range(SMALL_BATCH_SIZE):
                #     queue_dict[str(i+num)].append(all_data_file[i*SMALL_BATCH_SIZE+j])
            else:
                queue_dict[str(i+num)] = all_data_file[i*SMALL_BATCH_SIZE: i*SMALL_BATCH_SIZE+SMALL_BATCH_SIZE]


y_predict = [1, 2, 3, 4]
y = [2, 2, 2, 4]

dif_index = []
for i in range(len(y_predict)):
    if y[i] != y_predict[i]:
        dif_index.append(i) 

liebiao = [0,1,6,144,288,576,1152,2016,4320]
# liebiao = [0, 1, 2, 4, 7, 15]

# For Wrong Data
add_liebiao = liebiao[1:]
del_liebiao = [0]
for i in range(len(liebiao)):
    for j in range(i+1, len(liebiao)):
        del_liebiao.append(liebiao[j]-liebiao[i])
del_liebiao = list(set(del_liebiao))


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


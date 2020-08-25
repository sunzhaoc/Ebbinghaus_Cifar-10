'''
Description: 
Version: 1.0
Autor: Vicro
Date: 2020-08-22 10:33:57
LastEditors: Vicro
LastEditTime: 2020-08-22 11:07:14
'''
import os

NOW_BATCH = 0
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
SMALL_BATCH_SIZE = 3

for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
# for i in range(20):
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


y_predict = [1, 2]
y = [2, 2]

same_index = []
dif_index = []

for i in range(len(y_predict)):
    if y[i] == y_predict[i]:
        same_index.append(i)
    else:
        dif_index.append(i)

liebiao = [1,6,144,288,576,1152,2016,4320]
max_liebiao = liebiao[7]

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
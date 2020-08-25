'''
Description: 
Version: 1.0
Autor: Vicro
Date: 2020-08-21 18:57:31
LastEditors: Vicro
LastEditTime: 2020-08-21 20:52:16
'''
import os
SMALL_BATCH_SIZE = 5
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
""" Construct Queue Dict """
# for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
for i in range(20):
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

MAX_BATCH_SIZE = 3
for NOW_BATCH in range(99999999):
    queue_num = 0
    if len(queue_dict[str(NOW_BATCH)]) < MAX_BATCH_SIZE:
        for i in range(len(queue_dict)):
            for j in range(len(queue_dict[str(i+1)])):
                queue_dict[str(NOW_BATCH)] += [queue_dict[str(i+1)][0]]
                del queue_dict[str(i+1)][0]

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
        del queue_dict[str(NOW_BATCH)][-del_num:]

                
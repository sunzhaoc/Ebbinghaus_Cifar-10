'''
Description:  Fix Queue Length
Version: 1.0
Autor: Vicro
Date: 2020-08-21 18:57:31
LastEditors: Vicro
LastEditTime: 2020-08-23 21:26:51
'''
import os
SMALL_BATCH_SIZE = 2
all_data_file = os.listdir('Z:/STUDY/all_cifar') # Data Directory
""" Construct Queue Dict """
# for i in range(len(all_data_file)//SMALL_BATCH_SIZE):
liebiao = [0,1,6,144,288,576,1152,2016,4320]
for i in range(20):
    for num in liebiao:
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


MAX_BATCH_SIZE = 1
for NOW_BATCH in range(99999999):

    # Larger than Set
    if len(queue_dict[str(NOW_BATCH)]) < MAX_BATCH_SIZE:
        for i in range(NOW_BATCH+1, len(queue_dict)):
            for j in range(len(queue_dict[str(i)])):
                queue_dict[str(NOW_BATCH)] += [queue_dict[str(i)][0]]
                del queue_dict[str(i)][0]

                temp_dict = []
                [temp_dict.append(k) for k in queue_dict[str(NOW_BATCH)] if not k in temp_dict]
                queue_dict[str(NOW_BATCH)] = temp_dict

                if len(queue_dict[str(NOW_BATCH)]) >= MAX_BATCH_SIZE:
                    break
            if len(queue_dict[str(NOW_BATCH)]) >= MAX_BATCH_SIZE:
                break
        
        # Caculate Move Step
        move_step = 0
        for i in range(NOW_BATCH+1, len(queue_dict)):
            if len(queue_dict[str(i)]) == 0:
                move_step += 1
            else:
                break
        
        # Adjust Queue
        for i in range(NOW_BATCH+1, len(queue_dict)-move_step):
            queue_dict[str(i)] = queue_dict[str(i+move_step)]
        for i in range(len(queue_dict)-move_step, len(queue_dict)):
            del queue_dict[str(i)]
    
    # Small Than Set
    if len(queue_dict[str(NOW_BATCH)]) > MAX_BATCH_SIZE:
        del_num = len(queue_dict[str(NOW_BATCH)]) - MAX_BATCH_SIZE
        temp_dict = queue_dict[str(NOW_BATCH)][-del_num:]
        temp_dict += queue_dict[str(NOW_BATCH+1)]
        queue_dict[str(NOW_BATCH+1)] = temp_dict
        # Delete Same Data
        del queue_dict[str(NOW_BATCH)][-del_num:]
        # print(len(queue_dict[str(NOW_BATCH+1)]))
        queue_dict[str(NOW_BATCH+1)] = list(set(queue_dict[str(NOW_BATCH+1)]))
        # print(len(queue_dict[str(NOW_BATCH+1)]))

'''
Description: 
Version: 1.0
Autor: Vicro
Date: 2020-08-21 08:13:38
LastEditors: Vicro
LastEditTime: 2020-08-21 10:15:40
'''

# import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import csv
import codecs

input = open('ebbinghaus.log', 'r')
# input = open('test.txt', 'r')

rangeUpdateTime = []
out_dict = []

for line in input:
    line = line.split()
    # print(line)
    rangeUpdateTime.append(line[8])

for i in range(len(rangeUpdateTime)):
    out_dict.append(rangeUpdateTime[i][8:13])

# print(out_dict)

# CSV
# file_name = 'test.csv'
# def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
#     file_csv = codecs.open(file_name,'w+','utf-8')#追加
#     writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     for data in datas:
#         writer.writerow(data)
#     print("保存文件成功，处理结束")

# data_write_csv(file_name, out_dict)


# TXT
file_name = 'ebbinghaus22.txt'
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")



text_save(file_name, out_dict)
print("------------END-------------")
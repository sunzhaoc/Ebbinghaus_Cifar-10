'''
@Description: 将数据分为90%的训练集和10%的测试集
@Version: 1.0
@Autor: Vicro
@Date: 2020-07-24 20:42:56
@LastEditors: Vicro
@LastEditTime: 2020-07-24 21:27:41
https://blog.csdn.net/AugustMe/article/details/93628673
'''

import os
import shutil # 移动图片的库

def preprocess_data():
    data_file = os.listdir('./Dataset') # 读取所有图片的名字

    cat_file = list(filter(lambda x:x[:3]=='cat', data_file))
    dog_file = list(filter(lambda x:x[:3]=='dog', data_file))

    train_root = './train'
    val_root = './val'

    for i in range(len(cat_file)):
        print(i)
        pic_path = './Dataset/' + cat_file[i]

        if i < len(cat_file)*0.9:
            obj_path = './train/cat/' + cat_file[i]
        else:
            obj_path = './val/cat/' + cat_file[i]
        
        shutil.move(pic_path, obj_path)
    
    for j in range(len(dog_file)):
        print(j)
        pic_path = './Dataset/' + dog_file[j]

        if j < len(dog_file)*0.9:
            obj_path = './train/dog/' + dog_file[j]
        else:
            obj_path = './val/dog/' + dog_file[j]

        shutil.move(pic_path, obj_path)


if __name__ == '__main__':
    preprocess_data()
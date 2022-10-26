from PIL import Image
import numpy as np
import os

def TGSSalt_dataloader():
    # dataset: https://www.kaggle.com/c/tgs-salt-identification-challenge 
    # TGSSalt
    # file_list = ['datasets/TGSSalt/train/images', 'datasets/TGSSalt/train/masks', 'datasets/TGSSalt/test/images']
    # test에는 labeling 안된 데이터만 있어서 사용하지 않고자 함.
    file_list = ['datasets/TGSSalt/train/images', 'datasets/TGSSalt/train/masks']
    x_train, y_train, x_test = list(), list(), list()

    for i in range(len(file_list)):
        # print("file_list : ", i)
        for file_name in [file for file in os.listdir(file_list[i])]:
            # print("\tfile_name : ", file_name)
            if i == 0 or i == 2:
                x_data_ = np.array(Image.open(file_list[i] + "/" + file_name))
                if x_data_.shape[-1] == 4:    
                    x_data_ = x_data_[:, :, :-1]
                    
                if i == 0:
                    x_train += [x_data_]
                else:
                    x_test += [x_data_]
            else:
                y_data_ = np.array(Image.open(file_list[i] + "/" + file_name))
                y_train += [y_data_]
                
    x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test) 
    x_train = x_train.reshape(-1, 3, 101, 101)
    
    return x_train, y_train
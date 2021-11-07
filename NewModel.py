import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import os

#image_datas = glob('D:/trainingData/rm_0')

groups_folder_path = 'D:/trainingData/'#여기가 끼어들기 갯수 폴더이름 상위폴더 경로
class_name = ["rm_0", "rm_1", "rm_2"]#오토바이 갯수 폴더 이름
dic = {"rm_0":0, "rm_1":1, "rm_2":2}
#num_classes = len(class_name)


X = []
Y = []

for idex, categorie in enumerate(num_classes):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename)
           # img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            X.append(img / 256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)
'''
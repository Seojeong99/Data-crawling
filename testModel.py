#import keras.backend.tensorflow_backend as K
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os, re, glob
import cv2
import pickle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

categories = ["rm_0", "rm_1", "rm_2"]

def Dataization(img_path):
    image_w = 100
    image_h = 100
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
    return (img / 256)


src = []
name = []
test = []
image_dir = "D:/testData/"#테스트데이터 경로

for file in os.listdir(image_dir):
    if file.find('.jpg') != -1:
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
model = load_model('model_save/3.h5')
#time.sleep(10)
#loss, acc = model.evaluate(X_test, Y_test)
#print("\nLoss: {}, Acc : {}".format(loss,acc))
result = model.evaluate(test)
print(result)
predict = model.predict(test)
print(predict)
print(np.argmax(predict))
for i in range(len(test)):
    print(name[i])
    x = np.argmax(predict[i])
    if(x==0):
        print("오토바이가 0대입니다.")
    elif(x==1):
        print("오토바이가 1대입니다.")
    elif (x == 2):
        print("오토바이가 2대입니다.")
    print(predict[i])

'''
y_train_0=(Y_train==0)
y_test_0=(Y_test==0)
y_train_pred = cross_val_predict(model, X_train, y_train_0, cv=3)
cf = confusion_matrix(y_train_0, y_train_pred)
print(cf)
p = precision_score(y_train_0, y_train_pred)
print(p)
r = recall_score(y_train_0, y_train_pred)
print(r)
f1 = f1_score(y_train_0, y_train_pred)
print(f1)
'''
#import keras.backend.tensorflow_backend as K
import time
import sys
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

# 경로는 자신이 테스트해볼 파일의 경로로 바꿔주시면 됩니다!
from tensorflow.lite.python.schema_py_generated import np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

groups_folder_path = 'D:/trainingData/'#여기가 끼어들기 갯수 폴더이름 상위폴더 경로
categories = ["rm_0", "rm_1", "rm_2"]#오토바이 갯수 폴더 이름
num_classes = len(categories)

image_w = 100
image_h = 100
#image_w = 420
#image_h = 280

X = []
Y = []

for idex, categorie in enumerate(categories):
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



X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)


pickle.dump(xy, open("./img_data2.npy", 'wb'), protocol=4)

#np.save("./img_data.npy", xy)

X_train, X_test, Y_train, Y_test = np.load('./img_data2.npy', allow_pickle=True)

#num_classes = y_test.shape[1]

print("X_train data\n ", X_train)
print("y_train data\n ", Y_train)

model = keras.Sequential([

    Conv2D(32, kernel_size=(3, 3), padding='same',
           input_shape=X_train.shape[1:], activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),
    #conv2d
    Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Flatten(),
    Dense(64, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(num_classes, activation=tf.nn.softmax)

])


model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.summary()


history = model.fit(
    X_train, Y_train,
    batch_size=1, epochs=50,
    validation_data=(X_test, Y_test),
    #callbacks=[early_stopping],
    shuffle=True)

model.save('Gersang2.h5')

categories = ["rm_0", "rm_1", "rm_2"]

def Dataization(img_path):
    image_w = 100
    image_h = 100
    #image_w = 420
    #image_h = 280
    img = cv2.imread(img_path)
    #img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
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
model = load_model('Gersang2.h5')
#time.sleep(10)
loss, acc = model.evaluate(X_test, Y_test)
print("\nLoss: {}, Acc : {}".format(loss,acc))
predict = model.predict(test)
#for i in range(len(test)):
#    print(np.argmax(predict[i]))

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
def func(x):
    if x[0]==1:
        return 1
    else:
        return 0
totalsize = sys.getsizeof(Y_train)
rowsize = sys.getsizeof(Y_train[0])
rowCount = totalsize/rowsize
y_train_0 = []

for i in range(152):
    x = func(Y_train[i])
    y_train_0.append(x)

print(y_train_0)
#for i in range(152):
#    print(X_train[i])
y_train_pred = cross_val_predict(model, X_train, y_train_0, cv=3)
cf = confusion_matrix(y_train_0, y_train_pred)
print(cf)

'''
'''

#y_train_0 = list(map(func,Y_train))

#print(y_train_0)
#pst=lambda Y_train.find(1)1의 위치 반환
#print(Y_train[0][0])

#print(Y_train[0].find('0'))

#y_test_0 = (Y_test==)
y_train_pred = cross_val_predict(model, X_train, y_train_0, cv=3)
cf = confusion_matrix(y_train_0, y_train_pred)
print(cf)
'''
'''
#배열을 한줄한줄 읽어오면서 그 배열에 해당값이 있으면 1을 반환해서 list에 넣는다
y_train_0 = (Y_train==list(map(lambda 1이 첫번째에 있는것:,Y_train)))
#y_test_0 = (Y_test==)
y_train_pred = cross_val_predict(model, X_train, y_train_0, cv=3)
cf = confusion_matrix(y_train_0, y_train_pred)
#앞이 맞는것 뒤가 예측값
print(cf)
'''
'''
p = precision_score(y_train_0, y_train_pred)
print(p)
r = recall_score(y_train_0, y_train_pred)
print(r)
f1 = f1_score(y_train_0, y_train_pred)
print(f1)

'''


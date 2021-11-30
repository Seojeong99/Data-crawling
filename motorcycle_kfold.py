import tensorflow as tf
from focal_loss import BinaryFocalLoss


from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential

import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# class weight / 라벨 불균형 문제 해결을 위한 코드
def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

labels_dict = {0: 242, 1: 168, 2: 4}
dict_class_weights = create_class_weight(labels_dict)
print(dict_class_weights)

image_datas = glob('D:/trainingData/motorcycle/*/*.jpg')

class_name=["rm_0", "rm_1", "rm_2"]

dic={"rm_0":0, "rm_1":1, "rm_2":2}

X, Y= [], []

for i in image_datas:
    image = cv2.imread(i)
    #image = open(i)
    image = np.array(image)
    X.append(image)
    label = i.split('\\')[1]
    label = dic[label]
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

train_images, test_images, train_labels, test_labels2 = train_test_split(X, Y, test_size=0.1, random_state=42)

print(len(train_images), len(test_images))

train_labels = train_labels[..., tf.newaxis]
test_labels = test_labels2[..., tf.newaxis]

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# len_train = len(train_labels)
len_test = len(test_labels)

N_BATCH = 32

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


historys=[]
fold_acc = []
fold_loss = []
val_acc = 0

fold_var = 1

skf = StratifiedKFold(n_splits=5) # k-fold
for train, val in skf.split(train_images, train_labels):
    
    print('index_kf_train:', len(train_images[train]))
    print('index_kf_validation:', len(train_images[val]))
    
    # print(train_images[train].shape)
    # print(train_images[val].shape)
    # print(train_labels[train].shape)
    # print(train_labels[val].shape)
    
    x_train, x_val = train_images[train], train_images[val] # train_data
    y_train, y_val = train_labels[train], train_labels[val] # val_data
    
    x_trains = x_train.astype(np.float32) / 255
    x_vals = x_val.astype(np.float32) / 255

    y_trains = keras.utils.to_categorical(y_train)
    y_vals = keras.utils.to_categorical(y_val)
    
    len_train = len(y_trains)
    len_val = len(y_vals)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_trains, y_trains)).shuffle(buffer_size=len_train).batch(N_BATCH)#여기
    val_dataset = tf.data.Dataset.from_tensor_slices((x_vals, y_vals)).shuffle(buffer_size=len_val).batch(N_BATCH)
    
    model = Sequential()#(필터수, (행, 열) relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
    model.add(Conv2D(32, (3, 3), input_shape=train_images.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
    es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

    checkpoint_path = './model_save/'+str(fold_var)+".ckpt"
    mc = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True, save_weights_only=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    N_EPOCHS = 50
    
    history = model.fit(train_dataset, epochs=N_EPOCHS, validation_data=val_dataset, callbacks = [mc, es, reduce_lr], class_weight = dict_class_weights)
    
    historys.append(history)
    model.load_weights(checkpoint_path)
    model_path = "./model_save/" + str(fold_var) + ".h5"
    model.save(model_path)
    
    x_test = test_images.astype(np.float32) / 255
    test_label = keras.utils.to_categorical(test_labels)
    
    len_test = len(test_label)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_label)).shuffle(buffer_size=len_test).batch(N_BATCH)
    print(len(test_dataset))
    loss, acc = model.evaluate(test_dataset)
    fold_acc.append(acc)
    fold_loss.append(loss)

    print('loss :',loss, '\n Acc :', acc)
    
    fold_var += 1
    
    val_acc += acc
    y_pred = []
    predict = model.predict(test_dataset)
    for i in range(len(test_dataset)):
        y_pred.append(np.argmax(predict[i]))

    print(y_pred)
'''
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

    print(classification_report(test_label, y_pred))
    print('정확도 : {:.2f}'.format(accuracy_score(test_label, y_pred) * 100))
    print('정밀도 : {:.2f}'.format(precision_score(test_label, y_pred, average='weighted', zero_division=1) * 100))
    print('재현율 : {:.2f}'.format(recall_score(test_label, y_pred, average='weighted', zero_division=1) * 100))
    print('f1 score : {:.2f}'.format(f1_score(test_label, y_pred, average='weighted', zero_division=1) * 100))

print(val_acc /5)

print(fold_loss, fold_acc)

'''
'''
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
for history in historys:
    plot_graphs(history, 'loss')

print(test_labels2)
print(val_dataset)
y_pred=[]
predict = model.predict(val_dataset)
print(len(val_dataset))
for i in range(len(val_dataset)):
    print(predict[i])
    y_pred.append(np.argmax(predict[i]))#testdata가 왜 2개밖에 없을까


print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
print(classification_report(test_labels2, y_pred))
print('정확도 : {:.2f}'.format(accuracy_score(test_labels2, y_pred)*100))
print('정밀도 : {:.2f}'.format(precision_score(test_labels2, y_pred, average='weighted', zero_division=1)*100))
print('재현율 : {:.2f}'.format(recall_score(test_labels2, y_pred, average='weighted', zero_division=1)*100))
print('f1 score : {:.2f}'.format(f1_score(test_labels2, y_pred, average='weighted', zero_division=1)*100))
'''
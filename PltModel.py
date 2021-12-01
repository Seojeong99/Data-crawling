import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix



image_datas = glob('D:/trainingData/motorcycle/*/*.jpg')
class_name=["rm_0", "rm_1", "rm_2"]
dic={"rm_0":0, "rm_1":1, "rm_2":2}
X = []
Y = []
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



train_images, test_images, train_labels, test_label = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=44)

print(test_label)

train_labels = train_labels[..., tf.newaxis]
test_labels = test_label[..., tf.newaxis]

train_images.shape, train_labels.shape, test_images.shape, test_labels.shape

#print(len(train_images))
#print(len(test_images))
len_train = len(train_labels)
len_test = len(test_labels)
unique, counts = np.unique(np.reshape(train_labels, (len_train,)), axis=-1, return_counts=True) #여기
dict(zip(unique, counts))

unique, counts = np.unique(np.reshape(test_labels, (len_test,)), axis=-1, return_counts=True) #여기
dict(zip(unique, counts))

N_TRAIN = train_images.shape[0]
N_TEST = test_images.shape[0]

train_images = train_images.astype(np.float32)/255
test_images = test_images.astype(np.float32)/255

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

learning_rate = 0.01
N_EPOCHS = 1
N_BATCH = 1
N_CLASS = 3

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=len_train).batch(N_BATCH).repeat()#여기
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(buffer_size=len_test).batch(N_BATCH)

model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same',
           input_shape=train_images.shape[1:], activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),

    Flatten(),
    Dense(64, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(3, activation=tf.nn.softmax)

])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

steps_per_epoch = N_TRAIN//N_BATCH
validation_steps = N_TEST//N_BATCH
print(steps_per_epoch, validation_steps)

history = model.fit(train_dataset, epochs=N_EPOCHS, steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset, validation_steps=validation_steps)
model.save('motorcycle.h5')
model.evaluate(test_dataset)

'''
#결과 눈으로 보기
predict = model.predict(test_dataset)
for i in range(len(test_dataset)):
    print(test_label[i])
    print(np.argmax(predict[i]))

'''
y_pred=[]
predict = model.predict(test_dataset)
for i in range(len(test_dataset)):
    y_pred.append(np.argmax(predict[i]))

print(len(test_dataset))
print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
print(classification_report(test_label, y_pred))
print('정확도 : {:.2f}'.format(accuracy_score(test_label, y_pred)*100))
print('정밀도 : {:.2f}'.format(precision_score(test_label, y_pred, average='weighted', zero_division=1)*100))
print('재현율 : {:.2f}'.format(recall_score(test_label, y_pred, average='weighted', zero_division=1)*100))
print('f1 score : {:.2f}'.format(f1_score(test_label, y_pred, average='weighted', zero_division=1)*100))

#import keras.backend.tensorflow_backend as K
import time

from keras_preprocessing.image import ImageDataGenerator
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

# 경로는 자신이 테스트해볼 파일의 경로로 바꿔주시면 됩니다!
from tensorflow.lite.python.schema_py_generated import np

No_cut_in = os.path.join('D:/newSimpleTest/00')
Cut_in = os.path.join('D:/newSimpleTest/01')
T_Cut_in = os.path.join('D:/newSimpleTest/02')

print('total training rock images:', len(os.listdir(No_cut_in)))
print('total training paper images:', len(os.listdir(Cut_in)))
print('total training scissors images:', len(os.listdir(T_Cut_in)))

No_files = os.listdir(No_cut_in)
print(No_files[:10])

One_files = os.listdir(Cut_in)
print(One_files[:10])

Two_files = os.listdir(T_Cut_in)
print(Two_files[:10])

pic_index = 2

TRAINING_DIR = "D:/newSimpleTest/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


VALIDATION_DIR = 'D:/newSampleTestTest/'
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=1
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=1
)


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()



model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=3, steps_per_epoch=3, validation_data = validation_generator, verbose = 1,
                    validation_steps=3)

model.save("rps.h5")


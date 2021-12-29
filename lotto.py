import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
rows = np.loadtxt("D:\lotto.csv", delimiter=",", encoding="UTF-8-sig")
row_count = len(rows)
print(row_count)


def numbers2ohbin(numbers):
    ohbin = np.zeros(45)  # 45개의 빈 칸을 만듬

    for i in range(6):  # 여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i]) - 1] = 1  # 로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌

    return ohbin


def ohbin2numbers(ohbin):
    numbers = []

    for i in range(len(ohbin)):
        if ohbin[i] == 1.0:
            numbers.append(i + 1)

    return numbers

numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count-1]
y_samples = ohbins[1:row_count]



train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))

print("train: {0}, val: {1}, test: {2}".format(train_idx, val_idx, test_idx))

model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_loss = []
train_acc = []
val_loss = []
val_acc = []


for epoch in range(10):

    model.reset_states()

    batch_train_loss = []
    batch_train_acc = []

    for i in range(train_idx[0], train_idx[1]):
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)

        loss, acc = model.train_on_batch(xs, ys)  # 배치만큼 모델에 학습시킴

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)

        loss, acc = model.test_on_batch(xs, ys)  # 배치만큼 모델에 입력하여 나온 답을 정답과 비교함

        batch_val_loss.append(loss)
        batch_val_acc.append(acc)

    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))

    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))


xs = x_samples[-1].reshape(1, 1, 45)

ys_pred = model.predict_on_batch(xs)
sorted_nums = sorted(ys_pred[0], reverse=True)


print(np.where(ys_pred[0] == sorted_nums[0]))
print(np.where(ys_pred[0] == sorted_nums[1]))
print(np.where(ys_pred[0] == sorted_nums[2]))
print(np.where(ys_pred[0] == sorted_nums[3]))
print(np.where(ys_pred[0] == sorted_nums[4]))
print(np.where(ys_pred[0] == sorted_nums[5]))






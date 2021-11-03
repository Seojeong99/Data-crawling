import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

###########################
# 이미지 읽어서 데이터 준비하기
paths = glob.glob('D:/traingData/rm_0/*.jpg')

paths = np.random.permutation(paths)
독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
종속 = np.array([paths[i].split('/')[-2] for i in range(len(paths))])
print(독립.shape, 종속.shape)
'''
독립 = 독립.reshape(233, 400, 850, 1)
종속 = pd.get_dummies(종속)
print(독립.shape, 종속.shape)
'''
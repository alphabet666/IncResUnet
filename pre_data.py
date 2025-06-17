import pandas as pd
import numpy as np
import os

dir = r'C:\Users\86150\Desktop\COSMIC数据\data\train'
dir1 = r"C:\Users\86150\Desktop\COSMIC数据\data\train"
listdir = os.listdir(dir)
print(len(listdir))
train_x = np.zeros(1)
train1 = np.zeros(1)
train_y = np.zeros(1)
for i in listdir:
    print('正在读取' + i)
    path    = os.path.join(dir1 , i)
    csv     = pd.read_csv(path)
    csv     = np.array(csv)
    # csv     = np.diff(csv,0)
    #print(csv.shape)
    x_train = csv[:, 3]
    #train = csv[:, 0]
    #train = train.reshape((len(train), 1))
    print(len(x_train))
    x_train = x_train.reshape((len(x_train), 1))
    # print(x_train)
    # print('diff前',x_train.shape)
    x_train = np.diff(x_train, axis=0)
    # print(x_train)
    # print('diff后',x_train.shape)
    x_train = np.vstack((np.zeros(1), x_train))
    #print(x_train.shape,y_tr)
    y_train = csv[:,4]
    y_train = y_train.reshape((len(y_train), 1))
    #print(x_train.shape, y_train.shape)
    # y_train = np.diff(y_train, 0)
    train_x = np.vstack((train_x,x_train))
    train_y = np.vstack((train_y, y_train))
    #train1 = np.vstack((train1,train))

train_x=np.delete(train_x,0,0)
#train = np.delete(train1,0,0)
print(train_x.shape)
train_y=np.delete(train_y,0,0)
print(train_x.shape,train_y.shape)
#np.save(r"C:\Users\86150\Desktop\plasma-bubble-main\nc_x.npy",train_x)
#np.save(r"C:\Users\86150\Desktop\plasma-bubble-main\nc_y.npy",train_y)
np.save("train_x.npy",train_x)
np.save("train_y.npy",train_y)

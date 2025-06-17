from tensorflow.keras import layers
import tensorflow.keras as k
from tensorflow.keras.models import Model, load_model, Sequential
from keras import backend as K
from tensorflow.keras.layers import Layer

import tensorflow as tf
#from torch import nn
from tensorflow.keras import regularizers
import numpy as np
#import torch

class Unet_model(object):
    def __init__(self):
        self.windows = 1280
        self.final_act = 'sigmoid'
        #self.filter = [64, 128, 256, 512, 1024]
        #self.filter = [32, 64, 128, 256, 512]
        self.filter = [4, 8, 16, 32, 64]

    # .................................................................................................................
    
    def inceptionblock(self,planes,input,stride):
        residual = input

        b1 = layers.Conv1D(planes // 4,kernel_size=2,strides=stride,padding='same')(input)
        b1 = layers.LeakyReLU(alpha=0.2)(b1)
        b1 = layers.BatchNormalization()(b1)

        b2 = layers.Conv1D(planes // 4,kernel_size=3,strides=stride,padding='same')(input)
        b2 = layers.LeakyReLU(alpha=0.2)(b2)
        b2 = layers.BatchNormalization()(b2)

        b3 = layers.Conv1D(planes // 4, kernel_size=4, strides=stride, padding='same')(input)
        b3 = layers.LeakyReLU(alpha=0.2)(b3)
        b3 = layers.BatchNormalization()(b3)

        b4 = layers.Conv1D(planes // 4, kernel_size=5, strides=stride, padding='same')(input)
        b4 = layers.LeakyReLU(alpha=0.2)(b4)
        b4 = layers.BatchNormalization()(b4)

        b =layers.concatenate([b1,b2,b3,b4],axis=2)
        out = b+residual
        out = layers.ReLU()(out)
        return out

    def ResIncUnet(self):
        inputs = layers.Input((self.windows, 1))

        x1 = layers.Conv1D(filters=4,kernel_size=1,strides=1,padding='same',activation='relu')(inputs)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.LeakyReLU(alpha=0.2)(x1)
        x1 = self.inceptionblock(4,x1,stride=1)

        x2 = layers.MaxPooling1D()(x1)
        x2 = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same', activation='relu')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.LeakyReLU(alpha=0.2)(x2)
        x2 = self.inceptionblock(8, x2, stride=1)
        #x2 = layers.Dropout(0.5)(x2)

        x3 = layers.MaxPooling1D()(x2)
        x3 = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(x3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.LeakyReLU(alpha=0.2)(x3)
        x3 = self.inceptionblock(16, x3, stride=1)
        #x3 = layers.Dropout(0.5)(x3)

        x4 = layers.MaxPooling1D()(x3)
        x4 = layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(x4)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.LeakyReLU(alpha=0.2)(x4)
        x4 = self.inceptionblock(32, x4, stride=1)
        #x4 = layers.Dropout(0.5)(x4)

        x5 = layers.MaxPooling1D()(x4)
        x5 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(x5)
        x5 = layers.BatchNormalization()(x5)
        x5 = layers.LeakyReLU(alpha=0.2)(x5)
        x5 = self.inceptionblock(64, x5, stride=1)
        #x5 = layers.Dropout(0.5)(x5)

        x4_up = layers.UpSampling1D()(x5)
        x4_up = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4_up)
        x4_up = x4_up + x4
        # x4_up = layers.concatenate([x4_up,x4],axis=2)
        x4_up = layers.Conv1D(self.filter[3], kernel_size=1, padding='same', activation='relu')(x4_up)
        x4_up_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(x4_up)
        x4_up_ = layers.Conv1D(self.filter[3], kernel_size=3, padding='same', activation='relu')(
            x4_up + x4_up_)
        x4_up = x4_up + x4_up_

        x3_up = layers.UpSampling1D()(x4_up)
        x3_up = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3_up)
        x3_up = x3_up + x3
        x3_up = layers.Conv1D(self.filter[2], kernel_size=1, padding='same', activation='relu')(x3_up)
        x3_up_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(x3_up)
        x3_up_ = layers.Conv1D(self.filter[2], kernel_size=3, padding='same', activation='relu')(
            x3_up + x3_up_)
        x3_up = x3_up + x3_up_

        x2_up = layers.UpSampling1D()(x3_up)
        x2_up = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2_up)
        x2_up = x2_up + x2
        x2_up = layers.Conv1D(self.filter[1], kernel_size=1, padding='same', activation='relu')(x2_up)
        x2_up_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(x2_up)
        x2_up_ = layers.Conv1D(self.filter[1], kernel_size=3, padding='same', activation='relu')(
            x2_up + x2_up_)
        x2_up = x2_up + x2_up_

        x1_up = layers.UpSampling1D()(x2_up)
        x1_up = layers.Conv1D(self.filter[0], kernel_size=2, padding='same', activation='relu')(x1_up)
        x1_up = x1_up + x1
        x1_up = layers.Conv1D(self.filter[0], kernel_size=1, padding='same', activation='relu')(x1_up)
        x1_up_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(x1_up)
        x1_up_ = layers.Conv1D(self.filter[0], kernel_size=3, padding='same', activation='relu')(
            x1_up + x1_up_)
        x1_up = x1_up + x1_up_
        x1_up = layers.Dropout(0.5)(x1_up)

        y = layers.Conv1D(1, kernel_size=1, activation=self.final_act)(x1_up)

        model = Model(inputs, y)
        return model

if __name__ == '__main__':
    model = Unet_model().AttUnet()
    model.summary()
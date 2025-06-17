import os
import random
import numpy as np
import tensorflow as tf
import keras as K
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class _Data(object):
    def __init__(self):
        self.windows = 640

class generator_conv1d(object):
    def __init__(self, mode='train', batch_size=128, windows=640, ratio=0.7):
        self.mode = mode
        self.batch_size = batch_size
        self.windows = windows
        self.ratio = ratio

        print("[INFO] 正在加载训练和验证数据...")
        self.X_train = np.load(r"C:\Users\86150\Desktop\COSMIC数据\train_x.npy", allow_pickle=True).astype(np.float32)
        self.Y_train = np.load(r"C:\Users\86150\Desktop\COSMIC数据\train_y.npy", allow_pickle=True).astype(np.float32)
        self.X_val   = np.load(r"C:\Users\86150\Desktop\COSMIC数据\ROC_x.npy", allow_pickle=True).astype(np.float32)
        self.Y_val   = np.load(r"C:\Users\86150\Desktop\COSMIC数据\ROC_y.npy", allow_pickle=True).astype(np.float32)

        print("[INFO] 正在提取训练集索引...")
        self.index_train_1, self.index_train_0 = self.get_index(self.Y_train, desc="训练集")
        print("[INFO] 正在提取验证集索引...")
        self.index_val_1, self.index_val_0     = self.get_index(self.Y_val, desc="验证集")
        self.index_val = self.index_val_0 + self.index_val_1

        print(f"[INFO] 训练样本数量 - 正: {len(self.index_train_1)}，负: {len(self.index_train_0)}")
        print(f"[INFO] 验证样本数量 - 正: {len(self.index_val_1)}，负: {len(self.index_val_0)}")

        if len(self.index_val) == 0:
            raise ValueError("❌ 验证集索引为空！请检查 test_y.npy 是否含有正负标签。")

    def __iter__(self):
        if self.mode == 'train':
            return self.train_generator()
        else:
            return self.val_generator()

    def train_generator(self):
        input_train, output_train = [], []
        while True:
            x, y = self.get_train_data()
            input_train.append(x)
            output_train.append(y)
            if len(output_train) >= self.batch_size:
                yield np.array(input_train), np.array(output_train)
                input_train, output_train = [], []

    def val_generator(self):
        input_val, output_val = [], []
        while True:
            x, y = self.get_val_data()
            input_val.append(x)
            output_val.append(y)
            if len(output_val) >= self.batch_size:
                yield np.array(input_val), np.array(output_val)
                input_val, output_val = [], []

    def get_train_data(self):
        sampling = random.random()
        sampling_augment = random.random()

        if sampling < self.ratio:
            index = random.choice(self.index_train_1)
        else:
            index = random.choice(self.index_train_0)

        x = self.X_train[index - self.windows // 2:index + self.windows // 2, :]
        y = self.Y_train[index - self.windows // 2:index + self.windows // 2, :]

        # 数据增强（翻转+加噪声）
        if sampling_augment >= 1:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)
            x *= np.random.normal(1, 0.001, x.shape)
            x += np.random.normal(0, 0.001, x.shape)

        return x, y

    def get_val_data(self):
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            index = random.choice(self.index_val)
            if index - self.windows // 2 < 0 or index + self.windows // 2 > self.X_val.shape[0]:
                attempt += 1
                continue
            if (np.allclose(self.X_val[index - self.windows // 2, :], 0) or
                np.allclose(self.X_val[index + self.windows // 2, :], 0)):
                attempt += 1
                continue

            x = self.X_val[index - self.windows // 2:index + self.windows // 2, :]
            y = self.Y_val[index - self.windows // 2:index + self.windows // 2, :]
            return x, y

        print("⚠️ 警告：get_val_data 多次尝试失败，返回默认片段")
        index = self.windows
        x = self.X_val[index - self.windows // 2:index + self.windows // 2, :]
        y = self.Y_val[index - self.windows // 2:index + self.windows // 2, :]
        return x, y
    def get_index(self, file, desc="数据"):
        index_1, index_0 = [], []
        for i in tqdm(range(self.windows // 2, file.shape[0] - self.windows // 2), desc=f"[{desc}] 索引提取"):
            val = file[i, -1] if len(file.shape) > 1 else file[i]
            if np.isnan(val):  # 跳过 NaN
                continue
            if val == 1:
                index_1.append(i)
            else:
                index_0.append(i)

        return index_1, index_0
'''     def get_val_data(self):
        sampling = random.random()

        if sampling < self.ratio:
            index = random.choice(self.index_val_1)
        else:
            index = random.choice(self.index_val_0)

        x = self.X_val[index - self.windows // 2:index + self.windows // 2, :]
        y = self.Y_val[index - self.windows // 2:index + self.windows // 2, :]

        return x, y'''


if __name__ == '__main__':
    data = _Data()
    gen = generator_conv1d(mode='train')  # 会自动执行加载+索引+进度提示

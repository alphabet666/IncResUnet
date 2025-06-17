import os
import pandas as pd
import numpy as np
folder_path = r'C:\Users\86150\Desktop\COSMIC数据\data\all'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            # 先读取前几行判断是否是表头
            with open(file_path, 'r') as f:
                first_line = f.readline()

            # 判断是否是列名（只要不是纯数字就认为是表头）
            if any(c.isalpha() for c in first_line):
                df = pd.read_csv(file_path, header=0)  # 有列名自动去除
            else:
                df = pd.read_csv(file_path, header=None)  # 没有列名

            # 保留前两列，防止多余列影响
            df = df.iloc[:, :2]

            # 转为数值型，非法转化为NaN
            df = df.apply(pd.to_numeric, errors='coerce')

            # 删除任何有 NaN 的行
            df_cleaned = df.dropna().reset_index(drop=True)


            df_cleaned.to_csv(file_path, index=False, header=False)

            print(f"已清洗: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

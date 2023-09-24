import os

from utils.dataloader import MedicalImageDataset
from nets.model import HDB

# --------------------#
# 数据集路径
# --------------------#
Data_path = './data/Brain_DCM'
# --------------------#
# 训练
# --------------------#
train = True

with open(os.path.join(Data_path, "ImageSets/train.txt"), "r") as f:
    train_lines = f.readlines()
with open(os.path.join(Data_path, "ImageSets/val.txt"), "r") as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

dataset = MedicalImageDataset(train_lines, [256 * 192], train, Data_path)
print(num_train)
# # 获取 dataset[0] 的数据
data = dataset[0]
# print(data)
# import pandas as pd
# # 将数据转换成 DataFrame
# df = pd.DataFrame(data[1])
#
# # 将 DataFrame 保存为 CSV 文件
# df.to_csv('output.csv', index=False)
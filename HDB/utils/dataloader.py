from torch.utils.data import Dataset
import pydicom
import pandas as pd
import os
import torch
import numpy as np


class MedicalImageDataset(Dataset):
    def __init__(self, data_line, input_shape, train, dataset_path):
        self.data_line = data_line
        self.length = len(data_line)
        self.input_shape = input_shape
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_line = self.data_line[idx]
        name = data_line.split()[0]

        # -------------------------------#
        #   从文件中读取文件
        # -------------------------------#
        dcm = pydicom.dcmread(os.path.join(self.dataset_path, 'dcm', name + '.dcm'))
        # 获取高宽对应的数据
        pixel_data = dcm.pixel_array
        # 将pixel_data转换成DataFrame
        dcm_data = pd.DataFrame(pixel_data)

        MSB, LSB = self.split_MS(dcm_data)
        return MSB, LSB

    @staticmethod
    def split_MS(dcm_data):
        MSB = dcm_data.apply(lambda x: x // 2 ** 8).astype('uint8')
        LSB = dcm_data.apply(lambda x: x % 2 ** 8).astype('uint8')

        MSB_tensor = torch.tensor(MSB.values, dtype=torch.uint8)
        LSB_tensor = torch.tensor(LSB.values, dtype=torch.uint8)

        MSB_tensor = MSB_tensor.unsqueeze(0)
        LSB_tensor = LSB_tensor.unsqueeze(0)
        return MSB_tensor, LSB_tensor


def medical_image_dataset_collate(batch):
    MSB_batch = []
    LSB_batch = []

    for MSB, LSB in batch:
        MSB_batch.append(MSB)
        LSB_batch.append(LSB)


    # Convert lists to tensors
    MSB_batch = torch.from_numpy(np.array(MSB_batch)).type(torch.FloatTensor)
    LSB_batch = torch.from_numpy(np.array(MSB_batch)).type(torch.FloatTensor)
    return MSB_batch, LSB_batch

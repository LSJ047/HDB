import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   修改train_percent用于改变验证集的比例 9:1
#
#   当前该库将测试集当作验证集使用，不单独划分测试集
# -------------------------------------------------------#
trainval_percent = 1
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = 'E:/study/pythonProject/HDB/data/Brain_DCM'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    dcm_path = os.path.join(VOCdevkit_path, 'dcm')
    saveBasePath = os.path.join(VOCdevkit_path, 'ImageSets')

    temp_seg = os.listdir(dcm_path)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".dcm"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")




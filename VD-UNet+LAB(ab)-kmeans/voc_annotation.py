import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm



trainval_percent = 1
train_percent = 0.9

VOCdevkit_path = 'VOCdevkit'


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    try:
        temp_seg = os.listdir(segfilepath)
        total_seg = []
        for seg in temp_seg:
            if seg.endswith(".png"):
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
    except Exception as e:
        print(f"An error occurred: {e}")
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")

    classes_nums        = np.zeros([256], int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("No tag image %s was detected. Please check whether the file exists in the specific path and whether the suffix is png."%(png_file_name))

        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("label images %s shape:%s，It does not belong to grayscale or eight-digit color pictures, please carefully check the dataset format"%(name, str(np.shape(png))))


        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It was detected that the values of pixel points in the label only contain 0 and 255, and the data format was incorrect.")

    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("It is detected that the label contains only background pixels and the data format is incorrect. Please check the dataset format carefully.")

    print("The images in JPEGImages should be .jpg files, and the images in SegmentationClass should be .png files.")
    print("If the format is incorrect, refer to:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")
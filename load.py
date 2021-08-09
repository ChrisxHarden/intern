import json
import re
import numpy as np
import os
import cv2
import argparse
import sys
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.

    # Optional arguments.
    parser.add_argument(
        "--size",
        default=256,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    args = parser.parse_args()






img_shape = 256
class LOAD:
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]


    with open("/dev/shm/jj/modern_background_segmentation/utils/download/JudgmentItems.json", 'r',
              encoding='UTF-8') as f:
        judge_dict = json.load(f)

    for item in judge_dict:

        item['frameUri'] = item['frameUri'].replace("s3://mirror-converted-frames",
                                                    "/dev/shm/jj/modern_background_segmentation/utils/download")
        x_train_image_path = item['frameUri']
        if os.path.exists(x_train_image_path):
            img1 = cv2.imread(x_train_image_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (img_shape, img_shape))
            img1 = img_to_array(img1)
            img1 /= 255.0
            # x_train.append(img1)
        else:
            continue

        match_cross = re.match("s3://background-segmentation-image-datasets/cross_bu_silhouette_predicted_24e_od",
                               item["maskUri"])
        if match_cross:
            item['maskUri'] = item['maskUri'].replace("s3://background-segmentation-image-datasets",
                                                      "/dev/shm/jj/modern_background_segmentation/utils/download")  # cross
            y_train_image_path = item['maskUri']
            if os.path.exists(y_train_image_path):
                img2 = cv2.imread(y_train_image_path)  # imread函数打开图片

                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                img2 = cv2.resize(img2, (img_shape, img_shape))

                img2 = img_to_array(img2)

                img2 /= 255.0
                # y_train.append(img2)
            else:
                continue

        match_fix = re.match("s3://background-segmentation-pipeline/maskingFixups",
                             item["maskUri"])
        match_sample = re.match("s3://background-segmentation-pipeline/masksampler", item["maskUri"])
        if match_fix or match_sample:
            item['maskUri'] = item['maskUri'].replace("s3://background-segmentation-pipeline",
                                                      "/dev/shm/jj/modern_background_segmentation/utils/download")  # masksampler,fixup
            y_train_image_path = item['maskUri']
            if os.path.exists(y_train_image_path):
                img2 = cv2.imread(y_train_image_path)  # imread函数打开图片

                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                img2 = cv2.resize(img2, (img_shape, img_shape))

                img2 = img_to_array(img2)

                img2 /= 255.0
                # y_train.append(img2)
            else:
                continue

        match_appen = re.match("s3://background-segmentation-pipeline/masks/efficientnet", item['maskUri'])
        if match_appen:
            item['maskUri'] = item['maskUri'].replace("s3://background-segmentation-pipeline",
                                                      "/dev/shm/jj/modern_background_segmentation/utils/download/appen/outgoing")  # masks
            y_train_image_path = item['maskUri']
            if os.path.exists(y_train_image_path):
                img2 = cv2.imread(y_train_image_path)  # imread函数打开图片

                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

                img2 = cv2.resize(img2, (img_shape, img_shape))

                img2 = img_to_array(img2)
                for i in range(img_shape):
                    for j in range(img_shape):
                        if img2[i][j][0] >= 50:
                            img2[i][j][0] = 0
                        else:
                            img2[i][j][0] = 254

                img2 /= 255.0
                # y_train.append(img2)
            else:
                continue
        if os.path.exists(x_train_image_path) and os.path.exists(y_train_image_path):
            x_train.append(img1)
            print(1)

            y_train.append(img2)

        if len(x_train)>=1000:
            break #测试用


    num = len(x_train)
    num = int(num / 10)
    num *= 9
    print(num)
    print(len(x_train))
    x_val = x_train[num + 100:]
    y_val = y_train[num + 100:]
    x_test = x_train[num:num + 100]
    y_test_2 = y_train[num:num + 100]
    x_train = x_train[:num]
    y_train = y_train[:num]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    def get_x_train(self):
        return LOAD.x_train
    def get_y_train(self):
        return LOAD.y_train
    def get_x_val(self):
        return LOAD.x_val
    def get_y_val(self):
        return LOAD.y_val
    def get_x_test(self):
        return LOAD.x_test
    def get_y_test(self):
        return LOAD.y_test_2



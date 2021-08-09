import keras
import cv2
import os
import json
import re
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

with open("/dev/shm/jj/modern_background_segmentation/utils/download/JudgmentItems.json", 'r',
          encoding='UTF-8') as f:
    judge_dict = json.load(f)

i=0
for item in judge_dict:
    XP=False
    YP=False

    item['frameUri'] = item['frameUri'].replace("s3://mirror-converted-frames",
                                                "/dev/shm/jj/modern_background_segmentation/utils/download")
    x_train_image_path = item['frameUri']
    if os.path.exists(x_train_image_path):
        XP=True

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
            YP=True

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
            YP = True


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



            img2 = img_to_array(img2)
            for m in range(img2.shape[0]):
                for j in range(img2.shape[1]):
                    if img2[m][j][0] >= 50:
                        img2[m][j][0] = 0
                    else:
                        img2[m][j][0] = 254

            YP = True

        else:
            continue
    x_path="/dev/shm/jj/modern_background_segmentation/utils/download/train_data/imgs/"+str(i)+".png"
    y_path="/dev/shm/jj/modern_background_segmentation/utils/download/train_data/masks/"+str(i)+".png"
    if XP and YP:
        shutil.copyfile(x_train_image_path, x_path)

        cv2.imwrite(y_path, img2)
        i=i+1

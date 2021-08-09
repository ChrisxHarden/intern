import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.layers import Activation
import os
import cv2
import datetime
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import Conv2D, AveragePooling2D,Dense, Dropout, Flatten,Input,DepthwiseConv2D,Conv2DTranspose,UpSampling2D,BatchNormalization,Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import datetime
from keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,SGD
opt=Adam(learning_rate=0.003,amsgrad=True,decay=0.0005)
import argparse
import sys
from keras.regularizers import l1
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from random import randint
from keras.models import load_model
from PIL import Image


#from keras.metrics import MeanIoU
#metr=MeanIoU(num_classes=2)

#GPU控制代码
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3) #init use 30%
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



#gpus= tf.config.list_physical_devices('GPU') # copy来的解决方案
#print(gpus)# copy来的解决方案
#tf.config.experimental.set_memory_growth(gpus[0], True)# copy来的解决方案




#函数定义
def hardswish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def core_model(input_shape=(16, 16, 32)):
    main_inputs = Input(input_shape)

    x_1 = Conv2D(128, (1, 1), padding="same", activation=hardswish)(main_inputs)
    x_1 = DepthwiseConv2D((5, 5), padding="same", activation=hardswish)(x_1)
    x_2 = AveragePooling2D(pool_size=(16, 16))(x_1)
    x_2 = Conv2D(32, (1, 1), padding="valid", activation="relu")(x_2)
    x_2 = Conv2D(128, (1, 1), padding="valid", activation="sigmoid")(x_2)
    x_2 = x_2 * x_1
    x_2 = Conv2D(32, (1, 1), padding="same")(x_2)
    main_outputs = x_2 + main_inputs
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model


def connect(model1, model2):
    model = Model(model1.input, model2(model1.output))
    return model
#部分变量定义

'''''''''
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
'''



img_shape=256
main_input_shape=(img_shape,img_shape,3)
BATCHSIZE=32



#数据输入:
data_gen_args = dict(rescale=1./255,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.1)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 为 fit 和 flow 函数提供相同的种子和关键字参数
seed = 1
#image_datagen.fit(images, seed=seed)
#mask_datagen.fit(masks, seed=seed)

image_generator = image_datagen.flow_from_directory(
    '/dev/shm/jj/modern_background_segmentation/utils/download/train_data/1',
    target_size=(img_shape,img_shape),
    class_mode=None,
    color_mode='rgb',
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    '/dev/shm/jj/modern_background_segmentation/utils/download/train_data/2',
    class_mode=None,
    color_mode="grayscale",
    target_size=(img_shape,img_shape),
    seed=seed)

# 将生成器组合成一个产生图像和蒙版（mask）的生成器
train_generator = zip(image_generator, mask_generator)


#模型搭建
main_inputs = Input(main_input_shape)
x_0= Conv2D(16,(3,3) ,padding="same",activation=hardswish,strides=2)(main_inputs)
x_1=Conv2D(16,(1,1),padding="same",activation="relu")(x_0)
x_1=DepthwiseConv2D((3,3),padding="same",activation="relu",strides=2)(x_1)
x_2=AveragePooling2D(pool_size=(64,64))(x_1)
x_2=Conv2D(8,(1,1),padding="valid",activation="relu")(x_2)
x_2=Conv2D(16,(1,1),padding="valid",activation="sigmoid")(x_2)
x_1=x_2*x_1

x_1=Conv2D(16,(1,1),padding="same")(x_1)
x_2=Conv2D(72,(1,1),padding="same",activation="relu")(x_1)
x_2=DepthwiseConv2D((3,3),strides=2,padding="same",activation="relu")(x_2)
x_2=Conv2D(24,(1,1),padding="same")(x_2)

x_3=Conv2D(88,(1,1),padding="same",activation="relu")(x_2)
x_3=DepthwiseConv2D((3,3),padding="same",activation="relu")(x_3)
x_3=Conv2D(24,(1,1),padding="same")(x_3)
x_2=x_2+x_3

x_3=Conv2D(96,(1,1),padding="same",activation=hardswish)(x_2)
x_3=DepthwiseConv2D((5,5),strides=2,activation=hardswish,padding="same")(x_3)
x_4=AveragePooling2D(pool_size=(16,16))(x_3)
x_4=Conv2D(24,(1,1),activation="relu",padding="valid")(x_4)
x_4=Conv2D(96,(1,1),activation="sigmoid",padding="valid")(x_4)
x_3=x_4*x_3
x_3=Conv2D(32,(1,1) ,padding="same")(x_3)
model1=core_model()
model_all=Model(main_inputs,model1(x_3))
model2=core_model()
model_all=connect(model_all,model2)
model3=core_model()
model_all=connect(model_all,model3)
model4=core_model()
model_all=connect(model_all,model4)
x_4=AveragePooling2D(pool_size=(16,16))(model_all.output)
x_4=Conv2D(128,(1,1),activation="sigmoid")(x_4)

x_3=Conv2D(128,(1,1),activation="relu")(model_all.output)
x_3=x_4*x_3
x_3=tf.image.resize(x_3,(32,32))

x_3=Conv2D(24,(1,1),padding="valid")(x_3)
x_4=x_3+x_2

x_4=AveragePooling2D(pool_size=(32,32))(x_4)
x_4=Conv2D(24,(1,1),padding="valid",activation="relu")(x_4)
x_4=Conv2D(24,(1,1),padding="valid",activation="sigmoid")(x_4)
x_4=x_4*x_2
x_3=x_4+x_3

x_3=Conv2D(24,(1,1),activation="relu")(x_3)
x_4=DepthwiseConv2D((3,3),activation="relu",padding="same")(x_3)
x_3=x_3+x_4



x_3=tf.image.resize(x_3,(64,64))
x_3=Conv2D(16,(1,1))(x_3)
x_4=x_3+x_1
x_4=AveragePooling2D(pool_size=(64,64))(x_4)
x_4=Conv2D(16,(1,1),activation="relu")(x_4)
x_4=Conv2D(16,(1,1),activation="sigmoid")(x_4)
x_4=x_4*x_1
x_3=x_4+x_3


x_3=Conv2D(16,(1,1),activation="relu")(x_3)
x_4=DepthwiseConv2D((3,3),activation="relu",padding="same")(x_3)
x_3=x_3+x_4
x_3=tf.image.resize(x_3,(128,128))
x_3=Conv2D(16,(1,1))(x_3)
x_4=x_3+x_0
x_4=AveragePooling2D(pool_size=(128,128))(x_4)
x_4=Conv2D(16,(1,1),activation="relu")(x_4)
x_4=Conv2D(16,(1,1),activation="sigmoid")(x_4)
x_4=x_4*x_0
x_3=x_3+x_4
x_3=Conv2D(16,(1,1),activation="relu")(x_3)
x_4=DepthwiseConv2D((3,3),padding="same",activation="relu")(x_3)
x_3=x_3+x_4
x_3=Conv2DTranspose(1,(2,2),activation="sigmoid",strides=2,padding="same")(x_3)
#回调函数
#tensorboard
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#构建模型
model_trail=Model(model_all.input,x_3)
model_trail.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#检查点和Reduce_lr
checkpoint = ModelCheckpoint('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5',
    monitor='loss', save_weights_only=False,verbose=1,save_best_only=True, period=10)
reduce_lr=ReduceLROnPlateau(monitor='loss',patience=10,min_lr=0.000001,mode='auto',cooldown=2,verbose=1)


if os.path.exists('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5'):
    model_trail.load_weights('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5')
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")


history=model_trail.fit(
    train_generator,
    steps_per_epoch=56966/BATCHSIZE,
    epochs=300,
    callbacks=[tensorboard_callback,checkpoint]
)
#history = model_trail.fit(x_train, y_train, epochs=100,batch_size=25,callbacks=[tensorboard_callback,checkpoint],validation_data=(x_val,y_val))
#history = model_trail.fit(x_train, y_train, epochs=10,batch_size=25,callbacks=[tensorboard_callback,checkpoint],validation_data=(x_val,y_val))
x_test=[]
y_test_2=[]
for root, dirs, files in os.walk("/dev/shm/jj/modern_background_segmentation/utils/download/synthetic/imgs", topdown=False):
    for name in files:

        x_train_image_path = os.path.join(root, name)
        img1 = cv2.imread(x_train_image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (img_shape, img_shape))
        img1 = img_to_array(img1)
        img1 /= 255.0
        x_test.append(img1)



x_test=np.array(x_test)
for root, dirs, files in os.walk("/dev/shm/jj/modern_background_segmentation/utils/download/synthetic/masks",
                                 topdown=False):
    for name in files:
        x_train_image_path = os.path.join(root, name)
        img1 = cv2.imread(x_train_image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1 = cv2.resize(img1, (img_shape, img_shape))
        img1 = img_to_array(img1)
        img1 /= 255.0
        y_test_2.append(img1)
y_test_1=model_trail.predict(x_test)



logdir="logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

# Using the file writer, log the reshaped image.
with file_writer.as_default():
    images_x = np.reshape(x_test, (-1, 256, 256, 3))
    images_y1 = np.reshape(y_test_1, (-1, 256, 256, 1))
    images_y2 = np.reshape(y_test_2, (-1, 256, 256, 1))
    tf.summary.image("Training data", images_x,max_outputs=25,step=0)
    tf.summary.image("test",images_y1,max_outputs=25,step=0)
    tf.summary.image("comp", images_y2, max_outputs=25, step=0)

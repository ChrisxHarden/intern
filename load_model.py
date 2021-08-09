import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential,Model
#from keras.layers import Dense, Dropout, Flatten,Input,DepthwiseConv2D,Conv2DTranspose
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Dense, Dropout, Flatten,Input,DepthwiseConv2D,Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.layers import Activation
import os
import cv2
import datetime
import warnings
warnings.filterwarnings("ignore")
from keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
opt=Adam(learning_rate=0.003,amsgrad=True,decay=0.0005)

#控制代码
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3) #init use 30%
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


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
img_shape=256
main_input_shape=(img_shape,img_shape,3)
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


#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model_trail=Model(model_all.input,x_3)
model_trail.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

#checkpoint = ModelCheckpoint('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5',monitor='val_loss', save_weights_only=False,verbose=1,save_best_only=True, period=10)
#reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,mode='auto',cooldown=2)


if os.path.exists('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5'):
    model_trail.load_weights('/dev/shm/jj/modern_background_segmentation/utils/download/logs/trained_best_weights.h5')
    # 若成功加载前面保存的参数，输出下列信息
    print("model_loaded")

x_test=[]
for i in range(5):
    x_train_image_path="/dev/shm/jj/modern_background_segmentation/utils/download/train_data/imgs/"+str(56000+i)+".png"
    img1 = cv2.imread(x_train_image_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (img_shape, img_shape))
    img1 = img_to_array(img1)
    img1 /= 255.0
    x_test.append(img1)


x_test=np.array(x_test)
y_test_1=model_trail.predict(x_test)



logdir="logs/train_data/test_on_load/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)
with file_writer.as_default():
    images_x = np.reshape(x_test, (-1, 256, 256, 3))
    images_y1 = np.reshape(y_test_1, (-1, 256, 256, 1))
    tf.summary.image("Training data", images_x,max_outputs=25,step=0)
    tf.summary.image("test",images_y1,max_outputs=25,step=0)


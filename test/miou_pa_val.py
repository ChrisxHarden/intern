import numpy as np
import os
import tensorflow as tf
import cv2
import copy
import pandas as pd
from tensorflow.keras import backend as K
from keras.layers import Activation
import sys
import keras
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.generic_utils import CustomObjectScope
from keras.utils.generic_utils import get_custom_objects
import time
def hardswish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


#img_shape=256
get_custom_objects().update({'hardswish': Activation(hardswish)})

def pa_cal(x,y,img_shape):
    count=0
    #x[x>0.5] = 1
    sum=img_shape*img_shape
    for i in range(img_shape):
        for j in range(img_shape):
            #print(x.shape)
            #print(y.shape)
            if (x[i][j][0]>=0.5 and y[i][j][0]>=0.5) or (x[i][j][0]<=0.5 and y[i][j][0]<=0.5) :
                count=count+1

    return count/sum


def miou_cal(x,y,img_shape):
    count1=0
    count2=0
    for i in range(img_shape):
        for j in range(img_shape):
            if x[i][j][0]>=0.5 and y[i][j][0]>=0.5:
                count1=count1+1
            if x[i][j][0]>=0.5 or y[i][j][0]>=0.5:
                count2=count2+1


    return count1/count2

def load_predict_eval (model_path,model_size,x_test,y_test_1,ours=False):
    if ours==False:
        model = load_model(model_path, compile=False)
    else:
        model = load_model(model_path, compile=False,
                            custom_objects=get_custom_objects().update({'hardswish': Activation(hardswish)}))

    x_test_own=[]
    y_test_1_own=[]
    for img in x_test:
         img1 = cv2.resize(img, (model_size, model_size))
         img1 = img_to_array(img1)
         img1 /= 255.0
         x_test_own.append(img1)
    x_test_own = np.array(x_test_own)

    for img in y_test_1:
         img1 = cv2.resize(img, (model_size, model_size))
         img1 = img_to_array(img1)
         img1 /= 255.0
         y_test_1_own.append(img1)
    y_test_1_own = np.array(y_test_1_own)
    start_time=time.time()

    y_test=model.predict(x_test_own)
    end_time=time.time()
    used_time=end_time-start_time
    length=len(x_test)
    pa_sum=0
    miou_sum=0
    for i in range(length):
        a=y_test[i].reshape(model_size,model_size,1)
        pa_sum+=pa_cal(a,y_test_1_own[i],model_size)
        miou_sum+=miou_cal(a,y_test_1_own[i],model_size)

    val={'pa':pa_sum/length,'miou':miou_sum/length,'time':used_time}
    return val


##函数定义
x_test_path='C:\\Users\\enzhao\\photos\\imgs'
y_test_1_path='C:\\Users\\enzhao\\photos\\masks'
x_test=[]
y_test_1=[]
for root, dirs, files in os.walk(x_test_path, topdown=False):
    for name in files:

        x_train_image_path = os.path.join(root, name)
        img1 = cv2.imread(x_train_image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        #img1 = cv2.resize(img1, (img_shape, img_shape))
        #img1 = img_to_array(img1)
        #img1 /= 255.0
        x_test.append(img1)



#x_test=np.array(x_test)
for root, dirs, files in os.walk(y_test_1_path,
                                 topdown=False):
    for name in files:
        x_train_image_path = os.path.join(root, name)
        img1 = cv2.imread(x_train_image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        #img1 = cv2.resize(img1, (img_shape, img_shape))
        #img1 = img_to_array(img1)
        #img1 /= 255.0
        y_test_1.append(img1)
#y_test_1=np.array(y_test_1)


bilinear_seg=["C:\\Users\\enzhao\\codes\\Portrait-Segmentation-master\\models\\bilinear_seg\\bilinear_fin_munet.h5",128,False]

mnv3=["C:\\Users\\enzhao\\codes\\Portrait-Segmentation-master\\models\\mnv3_seg\\munet_mnv3_wm05.h5",224,False]

prisma_seg=["C:\\Users\\enzhao\\codes\\Portrait-Segmentation-master\\models\\prisma_seg\\prisma-net-15-0.08.hdf5",256,False]

transposeseg=["C:\\Users\\enzhao\\codes\\Portrait-Segmentation-master\\models\\transpose_seg\\deconv_fin_munet.h5",128,False]

our_model=["C:\\Users\\enzhao\\Desktop\\summary\\model\\trained_best_weights.h5",256,True]

#given_model=["C:\\Users\\enzhao\\Downloads\\model_float32.pb",256,True]
models=[bilinear_seg,mnv3,prisma_seg,transposeseg,our_model]

all_result=[]
for model in models:

    result=load_predict_eval(model_path=model[0],model_size=model[1],x_test=x_test,y_test_1=y_test_1,ours=model[2])
    all_result.append(result)
data=pd.DataFrame(all_result)
data.to_csv('C:\\Users\\enzhao\\Desktop\\result.csv')

'''''''''




interpreter = tf.lite.Interpreter(model_path="C:\\Users\\enzhao\\Downloads\\selfie.tflite")
interpreter.allocate_tensors()
x_test_own=[]
y_test_1_own=[]
for img in x_test:
    img1 = cv2.resize(img, (256, 256))
    img1 = img_to_array(img1)
    img1 /= 255.0
    x_test_own.append(img1)
x_test_own=np.array(x_test_own)

for img in y_test_1:
    img1 = cv2.resize(img, (256, 256))
    img1 = img_to_array(img1)
    img1 /= 255.0
    y_test_1_own.append(img1)
y_test_1_own = np.array(y_test_1_own)
b=np.expand_dims(x_test_own[0],axis=0)
interpreter.set_tensor(0,b)
y_test = interpreter.get_tensor(0)
y_test=np.array(y_test)
y_test = np.squeeze(y_test)
pa=pa_cal(y_test,y_test_1_own[0],256)
miou=miou_cal(y_test,y_test_1_own[0],256)
print(pa)
print(miou)

'''''''''





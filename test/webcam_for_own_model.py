import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import backend as K
from keras.layers import Activation
import sys
import keras
from keras.models import Model
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from keras.utils.generic_utils import get_custom_objects

def hardswish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

img_shape=256
get_custom_objects().update({'hardswish': Activation(hardswish)})



# Load the model and background image
model = load_model('C:\\Users\\enzhao\\Desktop\\summary\\\model\\trained_best_weights.h5', compile=False,custom_objects=get_custom_objects().update({'hardswish': Activation(hardswish)}))




# Load background image
bgd = cv2.resize(cv2.imread(sys.argv[1]), (513,513))
bgd = cv2.cvtColor(bgd, cv2.COLOR_BGR2RGB)/255.0

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret: 
      # Preprocess
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      simg = cv2.resize(img,(img_shape,img_shape))
      simg = simg.reshape((1,img_shape,img_shape,3))/255.0
      
      # Predict
      out=model.predict(simg)
      msk=np.float32((out>0.5)).reshape((img_shape,img_shape,1))


      # Post-process
      msk=cv2.GaussianBlur(msk,(5,5),1)
      img=cv2.resize(img, (513,513))/255.0
      msk=cv2.resize(msk, (513,513)).reshape((513,513,1))
      
      # Alpha blending
      frame = (img * msk) + (bgd * (1 - msk))
      #frame=out
      frame = np.uint8(frame*255.0)

      # Display the resulting frame
      cv2.imshow('portrait segmentation',frame[...,::-1])

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Sample run: python webcam.py test/beach.jpg

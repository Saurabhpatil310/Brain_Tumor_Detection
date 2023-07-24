import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask


app= Flask(__name__)


model=load_model('Braintumor10Epochs.h5')
print('model loaded. check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo ==1:
        return "Yes Brain Tumor"
    
def getResult(img):
    image=cv2.imread(img)
    image=Image.fromarray(img,'RGB')
    image= image.resize((64,64))
    image=np.array(image)
    input_img=np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


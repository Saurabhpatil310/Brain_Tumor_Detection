import cv2
import os
from PIL import Image
import numpy as np


image_directory='datasets/'

no_tumor_imges= os.listdir(image_directory+ 'no/')
yes_tumor_imges= os.listdir(image_directory+ 'yes/')

dataset=[]
label=[]

# print(no_tumor_imges)

# path='no0.jpg'
# print(path.split('.')[1])

#Iterate no_tumor_imges all images one by one through .+1 position(jpg)

for i , image_name in enumerate(no_tumor_imges):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_imges):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
    
# print(len(dataset))
# print(len(label))

#converting images into numpy array

dataset=np.array(dataset)
label=np.array(label)







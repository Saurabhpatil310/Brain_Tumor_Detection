import cv2
import os

image_directory='datasets/'

no_tumor_imges= os.listdir(image_directory+ 'no/')
yes_tumor_imges= os.listdir(image_directory+ 'yes/')



print(no_tumor_imges)

path='no0.jpg'
print(path.split('.')[1])


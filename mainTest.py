import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

#model selection

model= load_model('Braintumor10Epochs.h5')

#Getting images

images= cv2.imread('C:\\Users\\Saurabh Patil\\OneDrive\\Desktop\\CodeClause\\Brain\\BrainProject\\pred\\pred0.jpg')

# Resizing images

img = Image.fromarray(images)

img = img.resize((64, 64))  # Reassign the resized image back to the variable img

img = np.array(img)

input_img=np.expand_dims(img, axis=0)

# Getting the predicted probabilities

result = model.predict(input_img)

print(result)
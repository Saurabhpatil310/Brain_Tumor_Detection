import cv2
from keras.models import load_model
from PIL import Image

model= load_model('Braintumor10Epochs.h5')

images= cv2.imread('C:\\Users\\Saurabh Patil\\OneDrive\\Desktop\\CodeClause\\Brain\\BrainProject\\pred\\pred0.jpg')

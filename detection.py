#Required imports to get inference to work
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from picamera import PiCamera
import time

#loads the pre-trained model
model = load_model('MaskDetection/saved_model/saved_model')

#File output that is unique to each time the program is run
file_name = "MaskDetection/mask_test/img_" + str(time.time()) + ".jpg"

#initialization to the PiCamera
camera = PiCamera()
camera.resolution = (1280,720)
camera.vflip = False

#Shows a preview of the person before capturing their image.
print("Say Cheese")
time.sleep(1)
camera.start_preview()
time.sleep(2)
camera.capture(file_name)
camera.stop_preview()

#Prepares the image into an array so that an infrence can be made on the image
image = cv2.imread(file_name)
face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face = cv2.resize(face, (224, 224))
face = img_to_array(face)
face = preprocess_input(face)
face = np.expand_dims(face, axis=0)

#Running the inference and returns values for the probablility for a mask and without mask
(mask, withoutMask) = model.predict(face)[0]

if mask > withoutMask:
    print("Mask")
else:
    print("Without Mask")

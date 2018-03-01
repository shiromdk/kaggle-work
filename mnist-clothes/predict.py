import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd# -*- coding: utf-8 -*-
import cv2
import time
import os
from imutils.video import VideoStream
from threading import Thread
import imutils
import matplotlib.pyplot as plt

model = keras.models.load_model('model2.h5')




print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)
    image = cv2.resize(frame, (28, 28),interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = image.astype("float32") / 255.0
    plt.imshow(image)
    plt.show()
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
 
        
    label3 = " {}: {:.2f}% ".format( "Coat",prediction[4])
    label = "{}: {:.2f}%  {}: {:.2f}%  {}: {:.2f}%  {}: {:.2f}%   ".format("TSHIRT",
             prediction[0]* 100,"Trousers",prediction[1]*100,"Pullover",prediction[2],"Dress",prediction[3])
    label2 = "{}: {:.2f}%  {}: {:.2f}%  {}: {:.2f}%  {}: {:.2f}%  {}: {:.2f}%".format("Sandals",prediction[5],"Shirt",prediction[6],"Sneakers",prediction[7],"Bag",prediction[8],"Boots",prediction[9])
    frame = cv2.putText(frame, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.putText(frame, label2, (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.putText(frame, label3, (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF






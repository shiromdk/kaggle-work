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

model = keras.models.load_model('digit_model.h5')

df = pd.DataFrame(results)
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)
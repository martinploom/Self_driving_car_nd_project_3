# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []

# Load the csv data
# Run 1: CCW, mouse, 1 lap, new simulator

print("Opening files ...")
with open('driving_data/Run_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 
# Run 2: CW, mouse, 2 laps, new simulator
with open('driving_data/Run_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

"""       
with open('driving_data/Run_3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)        
"""
"""
print("Old file 0: ", lines[0][0])
print("Old file 1: ", lines[0][1])
print("Old file 2: ", lines[0][2])

print("New file 0: ", lines[3227][0])
print("New file 1: ", lines[3227][1])
print("New file 2: ", lines[3227][2])
"""

images = []
measurements = []
correction = 0.2 # Parameter to tune the right/left image correction

# Load the images and steering angles

print("Loading images and steering information ...")
for line in lines:
    
    # Create the steering measurements
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
        
    # Create the images
    source_path_center  = line[0]
    source_path_left    = line[1]
    source_path_right   = line[2]
    
    image_center    = cv2.imread(source_path_center)
    
    #image_left      = cv2.imread(source_path_left[1:])
    image_left      = cv2.imread(source_path_left)
    
    #image_right     = cv2.imread(source_path_right[1:])
    image_right     = cv2.imread(source_path_right)
    
    # Append the arrays
    measurements.append(steering_center)    
    images.append(image_center)
    measurements.append(steering_left)    
    images.append(image_left)
    measurements.append(steering_right)    
    images.append(image_right)
   
#print(measurements[0])
#path_center = lines[0][0]    
#path_left = lines[0][1]  
#print("PATH CENTER: ",  path_center)
#print(path_left)
#print(path_left[1:])
#plt.imshow(cv2.imread(path_left[1:]))
    
# Augment the data by flipping it along y axis and inverting the steering angle
# to double the training data and make it more symmetrical in regards steering
# to the left or right
    
print("Augmenting data ...")    
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
 
# Create training set
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# PREPROCESSING THE DATA

# Normalize and mean center the image so the data isn't from 0 - 255 but
# from -0.5 to 0.5
model.add(Lambda (lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Crop the lower and the upper part of the image to not confuse the model
# with unimportant data. 65 px are cut from upper part of the picture and
# 25 px are cut from the bottom of the picture and 0 is cut from left/right
model.add(Cropping2D(cropping = ((65,25), (0,0))))

# THE MODEL BUILD-UP

"""
# LeNet architecture
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
"""

# Even more powerful network by Autonomous driving team in NVidia
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Model uses Mean Square Error as this is not classification problem but 
# regression network to predict steering angle and we want to minimize the 
# error between the ground truth and the predicted steering measurement
# and the MSE is a good loss function for this
model.compile(loss='mse', optimizer='adam')

# Shuffle the data and separate 20 % for validation
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')
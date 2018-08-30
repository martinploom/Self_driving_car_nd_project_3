import csv
import cv2
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt

# Load the csv data as lines which reprsesent each frame
lines = []

# Run 1: CCW, mouse, 1 lap
print("Opening files ...")
with open('driving_data/Run_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 
# Run 2: CW, mouse, 2 laps
with open('driving_data/Run_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Run 5: CCW, mouse, 1 lap
with open('driving_data/Run_5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Run 6: CCW, mouse, only last righthand corner
with open('driving_data/Run_6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Run 7: CCW, mouse, only last righthand corner
with open('driving_data/Run_7/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
correction = 0.3 # Parameter to tune the right/left image correction

# Load the images and steering angles
print("Loading images and steering information ...")
for line in lines:
    
    # Load midcamera and create sidecamera steering measurements
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
        
    # Load the images paths
    source_path_center  = line[0]
    source_path_left    = line[1]
    source_path_right   = line[2]
    
    # Load the images
    image_center    = cv2.imread(source_path_center)    
    image_left      = cv2.imread(source_path_left)    
    image_right     = cv2.imread(source_path_right)
    
    # Append the measurements and images to the arrays
    measurements.append(steering_center)    
    images.append(image_center)
    measurements.append(steering_left)    
    images.append(image_left)
    measurements.append(steering_right)    
    images.append(image_right)
   
    
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
from keras.layers import Dropout
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
# The dropout didn't yield much improvement
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Model uses Mean Square Error as this is not classification problem but 
# regression network to predict steering angle and we want to minimize the 
# error between the ground truth and the predicted steering measurement
# and the MSE is a good loss function for this
model.compile(loss='mse', optimizer='adam')

# Shuffle the data and separate 20 % for validation
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 4)

# Print the keys contained in the history object
print(history_object.history.keys())
# Result: dict_keys(['loss', 'val_loss'])

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#Save the trained model
model.save('model.h5')
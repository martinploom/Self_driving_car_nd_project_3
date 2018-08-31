# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_material/cnn-architecture-624x890.png "NVIDIA Autonomous vehicle team's CNN architecture"
[image2]: ./report_material/center_2018_08_23_09_15_54_068.jpg "Midroad driving"
[image3]: ./report_material/left_2018_08_23_09_15_54_068.jpg "Left camera"
[image4]: ./report_material/right_2018_08_23_09_15_54_068.jpg "Right camera"
[image5]: ./report_material/center_flipped_2018_08_23_09_15_54_068.jpg "Midroad driving flipped"
[image6]: ./report_material/center_crop_2018_08_23_09_15_54_068.jpg "Midroad driving cropped"
[image7]: ./report_material/2_epochs_0,2steering.png "Validation vs training MSE"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I am using neural network developed by NVIDIA's Autonomous diriving team. The model consists of a convolutional neural network with 3x3 & 5x5 filter sizes and depths between 24 and 64 (model.py lines 125-136) 

The model includes RELU layers for all convolutional layers to introduce nonlinearity and the data is normalized and mean centered in the model with using a Keras lambda layer (code line 103). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer (model.py lines 133), but it didn't have positive effect on the performance in the simulator while driving in autonomous mode. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-39). Also the validation and training set MSE was checked and training epochs was selected so that both of them would drop til the last epoch. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 142).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used 2 CW laps and 1 CCW lap to train the initial network and start to improve from that based on problems on the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go step-by-step. Initially LeNet architecture was used and when the pipeline seemed to be working a model developed by the NVIDIA's Autonomous driving team was used. After implementing the NVIDIA model I tried to improve its performance by adding Dropout layer, but it didn't yield any better results so I gathered more training data in cases the model failed to execute as supposed to.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In order to detect the overfitting I monitored the test and validation set results and in case the validation results started to increase but the test set results kept decreasing I figured that the model might be overfitting and I cut down the epochs til the point I saw both of the set results were decreasing.
 
After every modification to the model I trained it for 5 epochs and when I pinpointed the number of epochs to which point the mean square errors were decreasing I stopped the training and reset the number of epochs and trained the model to test it on the track.

Finally when the model was performing well and I could improve the driving by just adjusting parameters I started to gather driving data from the situations where the model would fail (for example the last righthand corner at the track or where the car after the bridge has to keep on the tarmack and not go on dirt) til the car drove the full lap in autonomous mode.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 125-136) consisted of a
normalization
5 convolutional layers
4 fully connected layers

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To gather as good data as possible I did couple of practice laps while steering with the mouse the get the best steering angles. After I was used to it, I recorded one lap of clockwise driving and trained the initial model based on that.

Example of a image recorded from the center of the road:

![alt text][image2]

I then used images from the side cameras as well to get more data to train on and added steering angle correction to both images with opposite sign:

![alt text][image3]
![alt text][image4]


Then I recorded 2 laps of CW driving on the track and one additional CCW driving lap.

To augment the data sat, I also flipped images and angles to generate more training data and also balance the dataset so there would be equal amount of data for left and right hand corners. For example, here is an previously shown midroad driving image that has then been flipped:

![alt text][image5]

Before feeding the images into CNN I cropped the upper and lower part of the image to remove the noninformative part of the image so the model doesn't get confusing information. An example based on the previously shown center image can be seen in the following picture:

![alt text][image6]

For training the network I didn't use generator as I tried it out at some point and it seemed to be much slower than loading all the images at one go and there didn't seem to be any problems with this approach so I ditched the generator approach.

After the collection process, I had 36 504 number of data points. I then preprocessed this data by diving the RGB values by 255 to normalize it and then subtracted 0.5 to mean center the data and make it better for the network to train on.

I finally randomly shuffled the data set and put 20% of the data into a validation set, which mean that 7301 samples were used for validation. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as after that the training MSE kept decreasing but the validation set MSE started to rise which indicated overfitting and it is evidenced by this graph:

![alt text][image7]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
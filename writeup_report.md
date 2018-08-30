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
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

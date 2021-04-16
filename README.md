# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./images/NVIDIA.jpg "NVIDIA Model"

Introduction
---
The goal of this project is to build a ML model to simulate a car to run in autonomous mode using a [simulator](https://github.com/udacity/self-driving-car-sim) provided by Udacity. 

To run this project you'll also need the
[CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) (or you can use the lab environment provided by Udacity).

I used what I have learned about deep neural networks and convolutional neural networks to clone driving behavior. The framework used to train, validate and test the model is Keras. The model will output a steering angle to an autonomous vehicle.

The provided simulator was also used for data collection, in order to improve the behavior of the car.

Required Files
---
To meet specifications, the project will require submitting five files: 
* [model.py](model.py) (script used to create and train the model)
* [drive.py](drive.py) (script to drive the car - it's the original provided by the course)
* [model.h5](model.h5) (trained Keras model)
* a report writeup file (you're reading it!)
* [video.mp4](video.mp4) (video recording of the vehicle driving autonomously around the track for at least one full lap)

Quality of Code
---
The [model](model.h5) provided caon be used to successfully operate the simulation. In order to do so, after setting up the environment and the simulator (an operation described in their own repositories) you can run the model by using the following:

```sh
python drive.py model.h5
```

In order to improve memory efficiency the code uses Python generators to generate data for training rather than storing the training data in memory ([here](https://github.com/CollazzoD/CarND-Behavioral-Cloning-P3/blob/4f07c810cb01b1e1bd538ba4395118134fbf8100/model.py#L76)). 

Model Architecture and Training Strategy
---
The model used is based on the [NVIDIA Model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), depicted in the following image (found [here](https://github.com/CollazzoD/CarND-Behavioral-Cloning-P3/blob/4f07c810cb01b1e1bd538ba4395118134fbf8100/model.py#L51) in the code).

![alt text][image1]

Data is normalized and cropped inside the model before the "Input planes" indicated in the image above.

Data is splitted in train and validation sets and a dropout layer has been added after each of the last three fully-connected layers in order to reduce overfitting. The model uses an Adam optimizer and a Mean Squared Error (MSE) as loss function.
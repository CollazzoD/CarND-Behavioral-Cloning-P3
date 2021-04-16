# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./images/NVIDIA.jpg "NVIDIA Model"
[image2]: ./images/model_mean_squared_error_loss_5_epoch_dropout.png "Error Loss of Final Model"
[image3]: ./images/center_example.jpg "Example of Center Image"
[image4]: ./images/left_example.jpg "Example of Left Image"
[image5]: ./images/right_example.jpg "Example of Right Image"

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

Overview
---
First of all, I tried to use only the data provided by the course, which is divided in left, right and center images. 

Left                |  Center               |   Right
:------------------:|:---------------------:|:--------------:
![alt text][image4] |  ![alt text][image3]  |![alt text][image5]

Furthermore, I did not add the dropout layers and was using only the NVIDIA model as is. To stay low, I started with 3 epochs of training. This was not enough, as the car was going off road.

Then I proceeded to augment the dataset by flipping the images and multiplying the steering by a -1 factor (see [here](https://github.com/CollazzoD/CarND-Behavioral-Cloning-P3/blob/4f07c810cb01b1e1bd538ba4395118134fbf8100/model.py#L93) in the code).

After augmenting the dataset the car responded much better, but in some places of the circuit it was driving off track or in the wall. This behavior was most present on the bridge of the track.

To improve the response, I proceeded to further augment the dataset by recording other data. In particular:

* I recorded the car passing in the center of the bridge
* I took all the points in which the car was going off track and I recorded the car recovering and going back in the center

After adding all this data I obtained a [model](model_no_dropout.h5) able to do a lap of the track but that was driving a lot on the left of the road and not in the center (see this [video](video_no_dropout.mp4)).

After adding the three dropout layers, I was able to obtain the [model](model.h5) submitted, which drive much more in the center of the road (see this [video](video.mp4)), but only when I increased the number of epochs to 5. 

The final error loss obtained is showed in the following image

![alt text][image2]

At the end, I had the following:

* 106578 images in the dataset (center + left + right) * 2 because of flipping
* 85262 images for training (80% of the total dataset)
* 21316 images for validation (20% of the total dataset)

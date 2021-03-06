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

[model_arch]: ./md_assets/model_arch.png "Model Visualization"
[before_process]: ./md_assets/before_process.png "Image before processing"
[after_process]: ./md_assets/after_process.png "Image after processing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* processor.py containing the script for loading and processing data
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture
I use Nvidia's deep learning model for self driving describe in this [post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
It has 3 5x5 convolution layer with stride of 2x2,
followed by 2 3x3 convolution layer with stride of 1x1.
After flattening, it has three feed forward layers of size 100, 50, 10 before predicting the final output.
Here is a picture from the blog post:

![alt text][model_arch]

The implementation is in model.py.

### 2. Model Details
I use Exponential Linear Unit (ELU) as the activation function and
added dropout with keep rate of 0.5 for each feed forward layer to reduce overfitting.
I use Adam optimizer without manually tuning the learning rate.
The model is trained for 15 epochs.

### 3. Data Collection and Processing
The data set contains example data downloaded from Udacity,
as well as data generated by driving 5 rounds on track1 in simulator.
I pre-processed the data by cropping images , grayscale, and normalized it.
Here is an example before and after pre-processing (without normalize):

![alt text 1][before_process] ![alt text 2][after_process]

In addition, I add images from left camera with +0.2 on steering angle,
and images from right camera with -0.2 on steering angle.

All images are randomly shuffled and splitted into a training set containing 80% of the data,
and a validation set containing 20% of the data.

The implementation is in processor.py.

### 4. Results
After 15 epochs, the model has 0.0262 training loss and 0.0235 validation loss.

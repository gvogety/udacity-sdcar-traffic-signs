#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images-For-Report/TrafficSigns-6RandomPerClassImages.png "Class Visualization"
[image2]: ./Images-For-Report/ImageDistTraining.png "Training"
[image3]: ./Images-For-Report/ImageDistValidation.png "Validation"
[image4]: ./Images-For-Report/ImageDistTest.png "Testing"
[image5]: ./Images-For-Report/TestImages.png "Test Images  Downloaded From Web"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[tstimage1]: ./test-images/Image01.png "100 Speed"
[tstimage2]: ./test-images/Image02.png "30 Speed"
[tstimage3]: ./test-images/Image03.png "Do not enter"
[tstimage4]: ./test-images/Image04.png "Left or right"
[tstimage5]: ./test-images/Image05.png "Road work"
[tstimage6]: ./test-images/Image06.png "Straight or right"
[tstimage7]: ./test-images/Image07.png "Yield"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here are the Rubric Points

### Files Submitted
	 [Note Book](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)
	 [Html](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/Traffic_Sign_Classifier.html)
	 Set of 12 Images in [test-data](https://github.com/gvogety/udacity-sdcar-traffic-signs/tree/master/test-data)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Pandas library is to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes = 43

Pandas library is also used to calculate the distribution of various classes for each data set. This is used to generate the bar charts below. Its also worth noting that each set of images for a particular class are all together in the input data set. For example, all images of Class0 are together, followed by Class1, followed by Class2 etc. This fact is used in displaying the images below for further analysis.

####2. Include an exploratory visualization of the dataset.

For each class, 6 random images are shown. 

![alt text][image1]

It is also worth noting the image distribution for each set (Training, Validation and Testing). As can be seen some classes are well represented, while other not that much. For training set, any class with less than 250 images is color-coded Red (too few samples), while less than 500 is color-coded yellow (still few).


![alt text][image2]


![alt text][image3]


![alt text][image4]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because for the purpose of this project, where most of the image classification is based on shapes, rather than colors, it is not worth the additional processing needed for when 3(R,G,B) layers are used. In other words, if all 3 layers are used, it would have taken 3 times the computing without any additional improvement in the accuracy. When this code is ultimately used for traffic sign detection in a Self-Driving car, a video stream is being processed.. when computational simplicity is more important.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data. Normalizing the data helps in centering the features to a range that is of the same order of the weights and biases updated during training (gradient descent and backpropagation). Without centering, gradients can take time to stabilize (hill-climb is not smooth).

I did not attempt additional augmentation or enhacing the number of images, although in hind-sight I felt more data would have helped in achieving better accuracy.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I experimented with 3 convolutions and 3 flat(fully connected) layers as well as 3 and 2 respectively. A 3,3 approach was only marginally better w.r.t validation and test accuracy. Each layer used a RelU activation for non-linearity. I used **xavier** initialization for all weights although I observed only marginal improvement in validation and test accuracy. For the fully connected layers, I used dropout as additional regularization technique. I did experiment with various drop outs and settled down with a keep probability of 0.75 as others were only marginally different.. No dropout was significantly worse than with dropout.
My final model consisted of the following layers:

| Layer        			|     Description  		     					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution1 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution2 3x3		| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 					|
| Convolution3 3x3		| 1x1 stride, same padding, outputs 8x8x64 		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 					|
| Fully connected 1		| Input 1024 Output 512      					|
| Dropout				| Keep Probability=0.75							|
| Fully connected 2		| Input 512 Output 256      					|
| Dropout				| Keep Probability=0.75							|
| Fully connected 3		| Input 256 Output 128      					|
| Dropout				| Keep Probability=0.75							|
| Output/Logits			| Input 128 Output 43 (n_classes)				|
| Softmax				| 	     										|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, an Adam optimizer is used. I experimented with various learning rates and batch sizes. While they are all marginally different, a batch size of 128 and learning rate of 0.001 seems to give the best performance. I ended up going to 100 epochs as I could see upto 0.97 validation accuracy sometimes. With < 50 epochs, I could achieve 0.95. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Initially I started with 2 convolution layers and 1 Fully connected layer. With this, the validation accuracy was around .90. Then, I increased the architecture to 3 convolution layers and 3 fully connected layers. This improved validation accuracy to 0.92. Then I introduced Xavier initialization for the weights, with only marginal improvement. The biggest improvement came from introducing dropout to fully connected layers. With atleast 50 epochs, validation accuracy is consistently above 0.95.

Training and validation accuracies were calcuated after every epoch, while testing accuracy was calculated after training has been completed.
My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.97 
* test set accuracy of 0.95

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][tstimage1] ![alt text][tstimage2] ![alt text][tstimage3] 
![alt text][tstimage4] ![alt text][tstimage5]
![alt text][tstimage6] ![alt text][tstimage7]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(100km/h) 	|  Speed limit(100km/h)  						| 
| Speed limit(30km/h) 	|  Speed limit(30km/h)							|
| Do not enter			|  Do not enter									|
| Left or Right Turn  	|  "Untrained Image"			 				|
| Road work				|  Road work     								|
| Go Straight or Right	|  Go Straight or Right     					|
| Yield					|  Yield   										|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 86%. This compares favorably to the accuracy on the test set of 95%. The only image that the model has trouble with is the one it never trained on. With the dropout regularization in place, I am confident there is little overfitting. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



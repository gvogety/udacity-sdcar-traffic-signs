#**Traffic Sign Recognition** 


[//]: # (Image References)

[image1]: ./Images-For-Report/TrafficSigns-6RandomPerClassImages.png "Class Visualization"
[image2]: ./Images-For-Report/ImageDistTraining.png "Training"
[image3]: ./Images-For-Report/ImageDistValidation.png "Validation"
[image4]: ./Images-For-Report/ImageDistTest.png "Testing"
[image5]: ./Images-For-Report/7TestImages.png "Test Images  Downloaded From Web"
[image6]: ./Images-For-Report/TestImagesWithTop5Probabilities.png "Results on Sample Images"
[image6]: ./Images-For-Report/Grayscale.png "Grayscale"
[tstimage1]: ./test-data/Image01.png "100 Speed"
[tstimage2]: ./test-data/Image02.png "30 Speed"
[tstimage3]: ./test-data/Image03.png "Do not enter"
[tstimage4]: ./test-data/Image04.png "Left or right"
[tstimage5]: ./test-data/Image05.png "Road work"
[tstimage6]: ./test-data/Image06.png "Straight or right"
[tstimage7]: ./test-data/Image07.png "Yield"

---
### Writeup 

Here are the Rubric Points

### Files Submitted
* [Note Book](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb)
* [Html](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/Traffic_Sign_Classifier.html)
*  Set of 7 Images in [test-data](https://github.com/gvogety/udacity-sdcar-traffic-signs/tree/master/test-data)


### Data Set Summary & Exploration

##### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Pandas library is used to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes = 43

Pandas library is also used to calculate the distribution of various classes for each data set. This is used to generate the bar charts below. It is also worth noting that each set of images for a particular class are all together in the input data set. For example, all images of Class0 are together, followed by Class1, followed by Class2 etc. This fact is used in displaying the images below for further analysis.

##### Include an exploratory visualization of the dataset.

For each class, 6 random images are shown in the following examplei (for the complete list, please check the html version of the submission [here](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/Traffic_Sign_Classifier.html)). 

![alt text][image1]

It is also worth noting the image distribution for each set (Training, Validation and Testing). As can be seen some classes are well represented, while other not that much. For training set, any class with less than 250 images is color-coded Red (too few samples), while less than 500 is color-coded yellow (still few).


![alt text][image2]


![alt text][image3]


![alt text][image4]

#### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because for the purpose of this project, where most of the image classification is based on shapes, rather than colors, it is not worth the additional processing needed for when 3(R,G,B) layers are used. In other words, if all 3 layers are used, it would have taken upto 3 times the computing without any additional improvement in the accuracy. When this code is ultimately used for traffic sign detection in a Self-Driving car, a video stream is being processed.. when computational simplicity is more important.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image7]

As a last step, I normalized the image data. Normalizing the data helps in centering the features to a range that is of the same order as the weights and biases updated during training (gradient descent and backpropagation). Without centering, gradients can take time to stabilize (hill-climb is not smooth).

I did not attempt additional augmentation or enhacing the number of images, although in hind-sight I felt more data would have helped in achieving better accuracy.

##### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer        					|     Description  		     					| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         				| 32x32x1 Grayscale image						| 
| Convolution1 3x3, Dropout    	| 1x1 stride, same padding, Keep-prob 0.75, outputs 32x32x16 	|
| RELU							|												|
| Max pooling	      			| 2x2 stride,  outputs 16x16x16 				|
| Convolution2 3x3, Dropout		| 1x1 stride, same padding, Keep-prob 0.75, outputs 16x16x32 	|
| RELU							|												|
| Max pooling	      			| 2x2 stride,  outputs 8x8x32 					|
| Convolution3 3x3, Dropout		| 1x1 stride, same padding, Keep-prob 0.75, outputs 8x8x64 		|
| RELU							|												|
| Max pooling	      			| 2x2 stride,  outputs 4x4x64 					|
| Fully connected 1				| Input 1024 Output 512      					|
| Dropout						| Keep Probability=0.75							|
| Fully connected 2				| Input 512 Output 256      					|
| Dropout						| Keep Probability=0.75							|
| Fully connected 3				| Input 256 Output 128      					|
| Dropout						| Keep Probability=0.75							|
| Output/Logits					| Input 128 Output 43 (n_classes)				|
| Softmax						| 	     										|
 


##### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, an Adam optimizer is used. I experimented with various learning rates and batch sizes. While they are all marginally different, a batch size of 128 and learning rate of 0.001 seems to give the best performance. I ended up going to 100 epochs as I could see upto 0.97 validation accuracy sometimes. With < 50 epochs, I could achieve 0.96. 

##### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Initially I started with 2 convolution layers and 1 Fully connected layer. With this, the validation accuracy was around .90. Also taking lot more epochs to reach 0.90. To improve learning is each epoch, I made it deeper by increasing the architecture to 3 convolution layers and 3 fully connected layers. This improved validation accuracy to 0.92, but test accuracy (done on a larger set) was still suffering indicating overfitting. Then I introduced Xavier initialization for the weights, with only marginal improvement. The biggest improvement both in validation and test accuracy came from introducing dropout to fully connected layers. I used dropout as additional regularization technique. I did experiment with various drop outs and settled down with a keep probability of 0.75 as others were only marginally different.. "No dropouti" was significantly worse than with dropout. Dropout with a probability of 0.5 needed lot more training (epochs) to achieve similar accuracies. With this model, With atleast 50 epochs, validation accuracy is consistently above 0.95. Test accuracy is around 0.95

Few parameters that were tuned:
* Learning rate: Started with 0.01 and tried upto 0.0001. A learning rate of 0.01 caused too many fluctations in accuracy numbers between epochs while 0.0001 was taking too long to converge. Settled on 0.001 as the learning was improving and converging fast enough for the data set we had.
* Batch Size: Tried 64, 128 and 256.. 128 was converging fast enough. 
* Dropout: Experimented with various "keep probabilities". 0.5 was taking too long to converge whereas 0.75 was reasonable. 1.0 caused overfitting.
* Epochs: In general, the larger the number of epochs, the better. 100 epochs was getting validation accuracy to 0.97 and test accuracy to consistently above 0.95. Even though 15-20 epochs seem to get to 0.95 validation accuracy, both test accuracy and the accuracy on the downloaded internet images, were suffering. 50 epochs seemed to be a good middle-ground.

Training and validation accuracies were calcuated after every epoch, while testing accuracy was calculated after training has been completed (after all ecpochs).
My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.96 - 0.97 (depending on the number of epochs)
* test set accuracy of 0.95+

#### Test a Model on New Images

Here are seven German traffic signs that I found on the web [image5]:

Individual images can be found [here](https://github.com/gvogety/udacity-sdcar-traffic-signs/blob/master/test-images)

![alt text][tstimage1] ![alt text][tstimage2] ![alt text][tstimage3] 
![alt text][tstimage4] ![alt text][tstimage5]
![alt text][tstimage6] ![alt text][tstimage7]


##### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(100km/h) 	|  Speed limit(80km/h)  						| 
| Speed limit(30km/h) 	|  Speed limit(30km/h)							|
| Do not enter			|  Do not enter									|
| Left or Right Turn  	|  "Untrained Image"			 				|
| Road work				|  Road work     								|
| Go Straight or Right	|  Go Straight or Right     					|
| Yield					|  Yield   										|


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. This compares favorably to the accuracy on the test set of 95%  given that one of images that the model has trouble with is the one it never trained on. With the dropout regularization in place, I am confident there is little overfitting. After removing dropout from Convolution layers, I was able to take validation accuracy to 97% that resulted with 6 out 7 images to be correctly predicted.

Here is an example of the top-5 softmax probabilities and their analysis.

![top-5 probabilities][image6]

##### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The probabilities are pretty close to 1.0.. almost suggesting overfitting. If we look at the initial analysis of the images (random 6 for each class), some of the images are very bad quality. When I tried with images similar to them, softmax probabilities were a bit spread out just like the unsuccessful case below. In contrast, the images tried from the web are very clear and no obstructions or additional artifacts in the images (like additional signs around or text for other information (e.g. No entry between 4pm-6pm, etc.))

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the predictions:

		 Result 0  Prediction 5  (Speed limit (80km/h)) Actual 7  (Speed limit (100km/h))
		 Result 1  Prediction 1  (Speed limit (30km/h)) Actual 1  (Speed limit (30km/h))
		 Result 1  Prediction 17 (No entry            ) Actual 17 (No entry            )
		 Result 0  Prediction 14 (Stop                ) Actual 99 (Untrained           )
		 Result 1  Prediction 25 (Road work           ) Actual 25 (Road work           )
		 Result 1  Prediction 36 (Go straight or right) Actual 36 (Go straight or right)
		 Result 1  Prediction 13 (Yield               ) Actual 13 (Yield               )

Here is a complete list of top-5 probabilities for each image:

		Prediction: 5  , Probablity:  0.9612644910812378
		Prediction: 2  , Probablity:  0.027884675189852715
		Prediction: 7  , Probablity:  0.00634510163217783
		Prediction: 3  , Probablity:  0.002374258590862155
		Prediction: 8  , Probablity:  0.0011767109390348196

		Prediction: 1  , Probablity:  0.9999516010284424
		Prediction: 2  , Probablity:  4.845438525080681e-05
		Prediction: 4  , Probablity:  2.8739350454998203e-08
		Prediction: 5  , Probablity:  1.1620353984609366e-10
		Prediction: 15 , Probablity:  6.184037310008605e-11

		Prediction: 17 , Probablity:  1.0 
		Prediction: 14 , Probablity:  1.165679052046327e-25
		Prediction: 12 , Probablity:  5.764514759166911e-29
		Prediction: 10 , Probablity:  4.258925946452927e-29
		Prediction: 9  , Probablity:  1.0072247983382364e-29

		Prediction: 14 , Probablity:  0.5502846240997314
		Prediction: 13 , Probablity:  0.3339013159275055
		Prediction: 12 , Probablity:  0.060917921364307404
		Prediction: 1  , Probablity:  0.018204350024461746
		Prediction: 35 , Probablity:  0.01328835915774107

		Prediction: 25 , Probablity:  1.0 
		Prediction: 22 , Probablity:  4.771009715272224e-36
		Prediction: 31 , Probablity:  2.2502537270954726e-37
		Prediction: 0  , Probablity:  0.0 
		Prediction: 1  , Probablity:  0.0

		Prediction: 36 , Probablity:  1.0 
		Prediction: 38 , Probablity:  2.1336670494656595e-13
		Prediction: 25 , Probablity:  1.9826417525965756e-13
		Prediction: 35 , Probablity:  1.1698803004892705e-13
		Prediction: 20 , Probablity:  6.20880435822993e-14

		Prediction: 13 , Probablity:  1.0 
		Prediction: 1  , Probablity:  1.0256778684466162e-35
		Prediction: 12 , Probablity:  3.442555139594995e-37
		Prediction: 0  , Probablity:  0.0 
		Prediction: 2  , Probablity:  0.0


For the first image, the model could not predict the sign. First choice has very high probability whereas the correct choice has very low probability. One of the reasons could be the quality of the images used for training the correct choice. Also the images are very close to each other 80 vs 100km/h. The top five soft max probabilities were(at least, they are all speed limits!! :-) )

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .961         			| Speed Limit (80km/h)   						| 
| .027     				| Speed Limit (50km/h) 							|
| .006					| Speed Limit (100km/h)							|
| .002	      			| Speed Limit (60km/h)			 				|
| .001				    | Speed Limit (120km/h)      					|


For the untrained image, the probabilities are as follows.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .55                  | Stop  											|
| .33                  | Yield                          |   
| .06                  | Priority road                         |   
| .02                  | Speed Limit (30km/h)                          |
| .01                  | Ahead Only                         | 


#### Enhancements

As mentioned before, I did not attempt augmentation of images and other attempts to increase the number of samples. Increasing the size of labelled data would improve training and hence testing accuracy.


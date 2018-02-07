# **Traffic Sign Recognition** 

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

[bar_chart]: 		./report_images/bar_chart.png "Classes distribution among the sets"
[bar_chart_train]: 	./report_images/bar_chart_train.png "Classes distribution in the training set"
[bar_chart_train_aug]: 	./report_images/bar_chart_train_aug.png "Classes distribution in the training set after balancing"
[aug_data]:		./report_images/aug_data.png "Augmented data"
[aug_data_gray]:	./report_images/aug_data_gray.png "Augmented data in grayscale"
[architecture]: 	./report_images/Sermanet.jpeg "Sermanet architecture"
[cm_valid]: 		./report_images/cm_valid.png "Confusion Matrix for validation set"
[cm_test]: 		./report_images/cm_test.png "Confusion Matrix for test set"
[classes_sample]: 	./report_images/classes_sample.png "Classes sample"
[mean_20]: 		./report_images/mean_20.png "Mean 20"
[tr_img]: 		./report_images/augmentation/translated.png "Translated image"
[rot_img]: 		./report_images/augmentation/rotated.png "Rotated image"
[sh_img]: 		./report_images/augmentation/sheared.png "Sheared image"
[br_img]: 		./report_images/augmentation/brightness.png "Brightness modified image"
[aug_img]: 		./report_images/augmentation/augmented.png "Final augmented image"
[5_new]: 		./report_images/5_new.png "5 new images"
[5_new_pred]: 		./report_images/5_new_pred.png "5 new images with prediction"
[top_5]: 		./report_images/top_5.png "Top_5 predictions"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

This is my report for the second project of the Udacity Self Driving Car Nanodegree.
Here is a link to my [project code](https://github.com/iraadit/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used python functions and the pandas and numpy libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is of 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3), meaning that the images are 32x32px color images
* The number of unique classes/labels in the data set is 43

Each of the picture is a photo of a traffic sign belonging to one of 43 classes, listed in the file _signnames.csv_.

Here is a sample of the different classes :

![alt text][classes_sample]

We can see high differences in illumination in the different samples.

I calculated the mean images of each images, to see how the signs were different among a same class. It shows that the samples are very similar, as we can easily recognize the traffic sign in each of the mean images.
Here is an example for the Speed limit (20km/h)

![alt text][mean_20]


### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes in the data are distributed among the 3 different sets.
We can see that the different classes have been distributed more or less in the same proportions among the 3 sets.
But that some classes are represented a lot more than others, our dataset being therefore very unbalanced.

![alt text][bar_chart]

Augmenting the training data to avoid to have a so large difference could be a pre-processing step helping to better train the model.

The images also differ significantly in terms of contrast and brightness, so it could be useful to apply some kind of histogram equalization to help the feature extraction.


### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to generate additional data because I noted the the training data was unbalanced, I decided to balance it.
It certainly isn’t the best thing to do as our validation and test sets are unbalanced in the same way, but it would permit to perform better on arbitrary test images. 

![alt text][bar_chart_train]

The maximum number of samples for any label in the train set was of 2010, so I balanced the dataset by duplicating pictures of each class so that each class had 2010 samples.
For the duplicated images, I slightly modified them as explained below.

CNNs have built-in invariance to small translations, scaling and rotations. The training set doesn't seem to contain those deformations, so we will add those in our data augmentation.
To add more data to the the data set, I used the following techniques, with random parameters :
* Translation
* Rotation
* Shear
* Brightness modification

Augmenting the data is also helping to reduce overfitting, by incorporating real world features into our training set, such as varying lighting conditions, points of views,…

Here is an example of an original image and of the different augmentation techniques executed on it, the last one being the final augmented image, combining the different techniques:

Translated image :

![alt text][tr_img]

Rotated image : 

![alt text][rot_img]

Sheared image :

![alt text][sh_img]

Brightness modified image :

![alt text][br_img]

Augmented image, combining all of the above transformations :

![alt text][aug_img]


The difference between the original data set and the augmented data set is the following :
* Balanced dataset (same number of samples for each class)
* Images that are translated, rotated, sheared and with modified brightness

In this case, I kept the original dataset in the augmented dataset, without transforming it. So one of the class (the one that had the most samples at the start) has no transformations on its images, while other have a lot of it.
It could be better to create the augmentation of data when training, so that we don’t have to save the modified images.

Other pre-processing could be done, such as [Contrast Limited Adaptive Histogram Equalization](http://scikit-image.org/docs/dev/api/skimage.exposure.html) to. It has been added in my code, but I didn’t run it because it’s taking too much time to.

After balancing :

![alt text][bar_chart_train_aug]


Then, I decided to convert the images to grayscale because it lowers the number of weights to determine and that Sermanet and LeCun wrote in their paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that the color channels didn’t seem to improve the network a lot (they even had better results with gray images).
I will therefore employ the **Y** channel of the **YUV** OpenCV conversion of the color images.

Here is an example of modified traffic sign images before and after grayscaling.

![alt text][aug_data]

![alt text][aug_data_gray]

As a last step, I normalized the image data because it helps the weights to converge easier.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Model

My final model is adapted from the Sermanet/LeCunn traffic sign classification journal article [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), with different weights (I trained the network on my portable computer, so I was limited due to the compute time. I will use Amazon EC2 for the next projects).

The model consists of the following layers:

| Layer         	|     Description	        						| 
|:---------------------:|:-----------------------------------------------------------------------------:| 
| Input         	| 32x32x1 GRAY (Y channel) image   						| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 32x32x12 					|
| RELU			|										|
| Max pooling	A      	| 2x2 stride,  outputs 16x16x12 						|
| Convolution 5x5	| 1x1 stride, VALID padding, outputs 16x16x24    				|		
| RELU			|										|
| Max pooling	 B     	| 2x2 stride,  outputs 8x8x24 							|
| ROUTE			| Flatten layers from A max-pooled (8x8x12 -> 768) and B (8x8x24 -> 1536)	|
| ROUTE			| Concatenation of A and max pooled B, output 2304         			|	
| Fully connected	| output 400        								|
| RELU			|										|
| Dropout		| keep_prob = 0.5								|
| Fully connected	| output 43 (number of classes)       						|
| Softmax		|         									|
|			|										|
|			|										|
 
![alt text][architecture]

It didn't feel it was necessary to add more convolutional layers as there is a low statistical invariance between the pictures we work on, as most of them are already centered and cropped around the sign. With the augmented data I created, it could help though.

Using SAME padding instead of VALID padding doesn’t seem to improve the performance here, so I kept VALID padding as it lead to less weights to determine.

To train the model, I used the Adam optimizer (already implemented in the LeNet lab). The final settings used were:
* batch size: 128
* epochs: 30
* learning rate: 0.001
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.5

After many tests, the learning rate of 0.001 seemed to learn fast enough without getting stuck in a local minimum.
It could be useful to use a learning rate decay, but it is not the case here.
I trained during 30 epochs, but the validation accuracy wasn’t really going up since a moment.

It tried to use regularization on the weights, but with bad results. I should investigate it.

#### Approach

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.968 
* test set accuracy of 0.948

As we are classifying images, a CNN is a good choice.
I started by using the LeNet model from the Lab.
It wasn’t going up 93% on validation dataset.
From there, I read the Sermanet/LeCun article and decided to try to implement it.
I decided to try to use the Sermanet model, because it was coming from an article on Traffic Signs classification and had shown to have good results for that application.
I added dropout out the first fully connected layer, to avoid overfitting on the training data.
I tuned the weights and biases and the dropout rate, but kept the other parameters as is (mu, sigma, batch size and learning rate) because they seemed to work fine.
I’ve not had results similar to those of the Sermanet paper, but they use more features at each stage of layer.
The 5x5 convolution layers could be changed to 2 3x3 convolution layers each, as 2 3x3 conv layers cover the same surface then 1 5x5 conv layer while using less weights.

I know that new networks have shown better result, as DenseNet by example, but I’ve not had the occasion to try to implement it.

#### Other metrics
I also outputted other metrics for the validation and test sets.

##### Metrics for validation

| Metric | Value |
|---------------------|---------------------------------------------| 
| Precision | 0.961193997481 |
| Recall | 0.958606496862 |
| f1_score | 0.958028557376 |
| accuracy_score | 0.968253968254 |

![alt text][cm_valid]

##### Metrics for test

| Metric | Value |
|---------------------|---------------------------------------------| 
| Precision | 0.921776605689 |
| Recall | 0.933149713654 |
| f1_score | 0.924613512108 |
| accuracy_score | 0.947901821061 |

![alt text][cm_test]

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][5_new]

The second image might be difficult to classify because there are graffitis on the sign.

The fourth image might be difficult to classify because it belongs to a traffic sign that isn’t present in the training set (130km/h), we will see if it is still recognized as a Speed limit sign.

The fifth image might be difficult to classify because the picture is taken from below.

#### Predictions

Here are the results of the prediction:

![alt text][5_new_pred]

| Image			        |     Prediction	        	| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		| General Caution   			| 
| No entry    			| No entry 				|
| Priority road			| Priority road				|
| Speed limit (130km/h)	      	| __Speed limit (20km/h)__		|
| Yield				| Yield      				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.948. The model generalizes well to other images (out of the original dataset).

The second image was correctly guessed, even with a graffiti on the sign.

The only bad guess is from a Traffic Sign class that was not among the training set classes, i.e. from Speed limit (130km/h) sign. However, it has been recognized as a Speed limit (20km/h) sign, showing that our network was able to determine it was a Speed limit sign.

#### Top_k

The code for making predictions on my final model is located in the last cells of the Ipython notebook.

![alt text][top_5]

For all the images except the second, the model is very sure of its results, with certainty at 1 or near it.

For the second sign, the highest score is of 0.8396 for No entry, which is a good guess, seeing that the sign have been modified by a graffiti.

For the fourth sign, the top_5 results are all of Speed limits signs, showing that even if we didn’t train our network with Speed limit (130km/h) signs, it is still able to recognize that this sign is a Speed limit. However, I don’t get why the model is so sure of itself, and I imagine that parameter tuning could help for that, but I don’t know what I should change.

##### Image 1
General caution, certainty: 1.0000000000

Speed limit (20km/h), certainty: 0.0000000000

Speed limit (30km/h), certainty: 0.0000000000

Speed limit (50km/h), certainty: 0.0000000000

Speed limit (60km/h), certainty: 0.0000000000


##### Image 2
No entry, certainty: 0.8396092057

Keep right, certainty: 0.1053350940

Turn left ahead, certainty: 0.0371371917

Go straight or left, certainty: 0.0170528404

Speed limit (20km/h), certainty: 0.0004862924

##### Image 3
Priority road, certainty: 1.0000000000

Roundabout mandatory, certainty: 0.0000000000

Speed limit (120km/h), certainty: 0.0000000000

Ahead only, certainty: 0.0000000000

No vehicles, certainty: 0.0000000000

##### Image 4
Speed limit (20km/h), certainty: 0.9976807833

Speed limit (120km/h), certainty: 0.0021219393

Speed limit (30km/h), certainty: 0.0001971717

Speed limit (50km/h), certainty: 0.0000001375

Speed limit (70km/h), certainty: 0.0000000002

##### Image 5
Yield, certainty: 1.0000000000

Speed limit (20km/h), certainty: 0.0000000000

Speed limit (30km/h), certainty: 0.0000000000

Speed limit (50km/h), certainty: 0.0000000000

Speed limit (60km/h), certainty: 0.0000000000


## Improvements

I am confident I could improve the performance of this model even further with a couple of other interesting ideas I had, but I’m running out of time.
I could try :
* Regularization
* More convolutional or fully connected layers
* Other architecture (like DenseNet)
* Perform a (Local) Histogram Equalization as a pre-processing step, to distinguish the features of the signs more easily
* Augment data by using the horizontal or vertical properties of some signs (some symmetries on signs could help to augment other sign classes : Turn Left Ahead and Turn Right Ahead by example)

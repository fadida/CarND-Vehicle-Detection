**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_resources/dataset_car.png
[image2]: ./writeup_resources/dataset_nonecar.png
[image3]: ./writeup_resources/dataset_car_features.png
[image4]: ./writeup_resources/dataset_nonecar_features.png
[image5]: ./writeup_resources/windows.png
[image61]: ./writeup_resources/window_detection_heatmap_1.png
[image62]: ./writeup_resources/window_detection_heatmap_2.png
[image63]: ./writeup_resources/window_detection_heatmap_3.png
[image64]: ./writeup_resources/window_detection_heatmap_4.png
[image65]: ./writeup_resources/window_detection_heatmap_5.png
[image66]: ./writeup_resources/window_detection_heatmap_6.png
[image71]: ./writeup_resources/labels_1.png
[image72]: ./writeup_resources/labels_2.png
[image73]: ./writeup_resources/labels_3.png
[image74]: ./writeup_resources/labels_4.png
[image75]: ./writeup_resources/labels_5.png
[image76]: ./writeup_resources/labels_6.png
[image81]: ./writeup_resources/final_1.png
[image82]: ./writeup_resources/final_2.png
[image83]: ./writeup_resources/final_3.png
[image84]: ./writeup_resources/final_4.png
[image85]: ./writeup_resources/final_5.png
[image86]: ./writeup_resources/final_6.png
[image91]: ./output_images/test1.jpg
[image92]: ./output_images/test4.jpg
[image93]: ./output_images/test5.jpg
[video1]: ./output_images/project_video.mp4

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `FeatureExtractor` class at the `_get_hog_features()` method and `hog()` methods [lines 142-189].
The `_get_hog_features()` warps the `skimage.feature.hog()` method and the `hog()` method extracts the HOG features for the channel(s) configured in the `FeatureExtractor`.

The HOG feature extraction is the second action taken when extracting features (changing the color space is the first action). The feature extraction process is implemented in the `__call__()` method in the `FeatureExtractor`.   

I started by downloading the images and creating the dataset by shuffling all the images together (`VehicleDetector._download_dataset()` method).
The dataset consists of two labels, car and none-car:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and trained the classifier with them until I got to a satisfying result, in my case to 99.07% accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `sklearn.model_selection.StratifiedShuffleSplit` class which split my data into 80% training set and 20% testing set. All of the dataset features were extracted before the split by calling `FeatureExtractor.__call__()` method.
The training method code can be found on `VehicleDetector.train()` method [lines 382-428].

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search using different window sizes and only on the region of interest which is defined on `DetectionPipeline._calc_pipeline_properties()` [lines 482-483].

The windows are created once on pipeline first run, at the `DetectionPipeline._calc_pipeline_properties()`, the method parameter `window_size_range` dictates what is the minimum and maximum sized windows and also the size gaps between the windows.

The window building algorithm calculates the number of window sizes that it needs to create and the location within the region of interest each window needs to be drawn in, the rule is: smaller windows location will be higher then larger windows. [lines 486-530]

![alt text][image5]

Because the pipeline needs to scan different sized windows and extract features from them, the window searching method `VehicleDetector._search_windows()` implementation assumes the following:
* The window list is sorted by size.
* All windows are squares.

The window searching algorithm is the following:
* Calculate window size by calculating the window width (uses second assumption). [line 567]
* check if the region of interest was processed for this window size and if not - resize the image so that the window will be in size 64X64 and apply HOG on the scaled region of interest. (uses first assumption) [lines 564-576]
* Use the `FeatureExtractor.extract_hog_features()` in order to sample the HOG region of interest and return the corresponding feature vector and also re-calculate the window coordinates over the scaled image. [line 578].
* Use the calculated window to get the region in the relevant pixels in the scaled image and send it to the `FeatureExtractor` and then to the classifier. [lines 580-584].

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here an example of some test images:

![alt text][image91]
![alt text][image92]
![alt text][image93]


In order to optimize performance I implemented HOG sampling and resize per window size (as explained in the section above).
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter out false positives I used two algorithms:
* **heat map filtering** -
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap (`VehicleDetector._apply_to_heatmap()` [lines 613-623]) and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap .  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. (`VehicleDetector._detect_from_heatmap()` [lines 625-648])

* **distance filtering** - For each frame, I recorded the previous frame detections center positions (except for the first frame). Then the center positions of the current frame detections is calculated. The filter is calculating the distance from each detected center to each of the previous detected centers. If the distance is smaller then a set threshold the detection will pass the filter and if the distance is bigger the detection will be ignored. After that the list of all detections center positions is saved (including the discarded detections center positions). (`VehicleDetector._calc_detection_from_prev()` [lines 650-687])

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image61]
![alt text][image62]
![alt text][image63]
![alt text][image64]
![alt text][image65]
![alt text][image66]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image71]
![alt text][image72]
![alt text][image73]
![alt text][image74]
![alt text][image75]
![alt text][image76]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image81]
![alt text][image82]
![alt text][image83]
![alt text][image84]
![alt text][image85]
![alt text][image86]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The approach I took when building this pipeline is to create a pipeline the supports both still images and videos. I tried to implement it more elegantly then  the pipeline I implemented at the last project.

I wanted to keep the classifier as simple as possible and let it stay SVM instead of more complex classifiers like CNNs. That means that false positives detection falls into the pipeline implementation itself instead of the classifier.

There were two main issues I faced when writing this project:
* **HOG sampling in window searching** - The main problem I had was because the windows I used are in different sizes. At first I thought that I can calculate HOG on the original region of interest, extract the features I need and interpolate/decimate  the feature vector to match the required size. But this implementation is too complex as the interpolation need to be trilinear interpolation. I then thought of much simpler way which is the algorithm I implemented in the end.
* **False positives and jittering** - I faced a lot of false positives even after applying the heat map. Also, the squares around the detected cars jittered a lot.
I wanted to find a way to use past detections in order to minimize both of those problems. At first I tried to give an higher values in the heat map for detections that were close to previous detected vehicles, but the false positives amount wasn't reduced (because I didn't want to introduce a penalty in the heat map). In the end I thought filtering is more elegant and can reduce the false positives without hindering the detection of future vehicles.   

The pipeline can fail when other types of cars will be introduced because that will make the classifier more complex or make it cover more complex conditions.
Also, I think this pipeline will be weak against car reflections, like if the road was wet and cars were reflected on the road.

If I were going to pursue this project further, the first thing I would do is make this pipeline work faster. The sliding windows I used can be improved by selectively making some areas contain more windows then others. This window selection process needs to be adaptive. Maybe creating a list of enabled/disabled windows in order to still create the window list once but scan the image using less windows.
Anther thing I might add is truly tracking a car. What I mean by that is to look for a specific car and track it. This could help because if data will be recorded according to a specific car, I think it can make the system create wiser decisions when planning the drive. IE: if a specific car displays hazardous driving behavior, the system should keep distance from it or let that car gain some distance in order to avoid danger.    

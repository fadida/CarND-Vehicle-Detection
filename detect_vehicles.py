"""
This script purpose is to implement a
vehicle detection pipeline for still
images and videos.
"""
from moviepy.editor import VideoFileClip
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.utils import shuffle

from scipy.ndimage.measurements import label

from skimage.feature import hog

import os
import pickle
import matplotlib.image as mpimg
import cv2
import numpy as np
import pycurl
import zipfile
import random
import tqdm
import shutil


class FeatureExtractor:
    """
    This class implements feature extraction
    method and configuration in order to insure
    that the detector will get the same configuration
    """
    def __init__(self, color_space, hog_channels, hog_orient, hog_pix_per_cell, hog_cell_per_block, spatial_size,
                 color_hist_bins, color_hist_bins_range):
        if color_space != 'RGB':
            try:
                self._cvt_code = eval('cv2.COLOR_RGB2' + color_space)
            except:
                raise Exception('Invalid color space {}'.format(color_space))
        else:
            self._cvt_code = None

        # HOG parameters
        self._hog_channels = hog_channels
        self._hog_orient = hog_orient
        self._hog_pix_per_cell = hog_pix_per_cell
        self._hog_cell_per_block = hog_cell_per_block

        # Spatial parameters
        self._spatial_size = spatial_size

        # Color histogram parameters
        self._color_hist_bins = color_hist_bins
        self._color_hist_bins_range = color_hist_bins_range

        self._scaler = StandardScaler()

    def __call__(self, img: [np.ndarray, list], hog_features=True, spatial_features=True, color_features=True):

        # In order to generalize the code for single image & multiple
        # images, single images are turned into an array of one image.
        if isinstance(img, np.ndarray) and len(img.shape) in (2, 3):
            img = np.array([img])
            single_img_mode = True
        else:
            img = tqdm.tqdm(img)
            single_img_mode = False

        features = []
        for single_img in img:
            single_img = self._change_color_space(single_img)

            img_features = []

            if isinstance(hog_features, bool) and hog_features:
                img_features.append(self._hog(single_img))
            elif hog_features is np.ndarray:
                img_features.append(hog_features)

            if isinstance(spatial_features, bool) and spatial_features:
                img_features.append(self._spatial(single_img))
            elif spatial_features is np.ndarray:
                img_features.append(spatial_features)

            if isinstance(color_features, bool) and color_features:
                img_features.append(self._color_hist(single_img))
            elif color_features is np.ndarray:
                img_features.append(color_features)

            img_features = np.concatenate(img_features).astype(np.float64)
            features.append(img_features)

        features = np.array(features)
        if single_img_mode:
            features = features.reshape(1, -1)
        else:
            self._scaler.fit(features)

        scaled_features = self._scaler.transform(features)
        return scaled_features

    def _get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self._hog_orient,
                                      pixels_per_cell=(self._hog_pix_per_cell, self._hog_pix_per_cell),
                                      cells_per_block=(self._hog_cell_per_block, self._hog_cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self._hog_orient,
                           pixels_per_cell=(self._hog_pix_per_cell, self._hog_pix_per_cell),
                           cells_per_block=(self._hog_cell_per_block, self._hog_cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def _hog(self, img: np.ndarray):
        if self._hog_channels == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.extend(self._get_hog_features(img[:, :, channel], vis=False, feature_vec=True))
        else:
            hog_features = self._get_hog_features(img[:, :, self._hog_channels], vis=False, feature_vec=True)

        return np.array(hog_features)

    def _spatial(self, img: np.ndarray):
        return cv2.resize(img, self._spatial_size).ravel()

    def _color_hist(self, img: np.ndarray):
        hist_features = []
        for channel in range(img.shape[2]):
            hist = np.histogram(img[:, :, channel], bins=self._color_hist_bins, range=self._color_hist_bins_range)
            hist_features.append(hist[0])

        return np.array(np.concatenate(hist_features))

    def _change_color_space(self, img):
        if self._cvt_code is not None:
            return cv2.cvtColor(img, self._cvt_code)
        else:
            return np.copy(img)


class VehicleDetector:
    """
    This class implements the model
    that detects vehicles from image
    feature vectors
    """
    def __init__(self, dataset_pickle_path, cached_pickle_path, load_from_cache=True):
        self._dataset_pickle_path = dataset_pickle_path
        self._cached_pickle_path = cached_pickle_path

        self._dataset = None
        self._feature_extractor = None

        if load_from_cache and os.path.exists(cached_pickle_path):
            print('Found cached classifier. Loading classifier')
            with open(cached_pickle_path, 'rb') as f:
                cache = pickle.load(f)
            self._cls = cache['cls']
            self._feature_extractor = cache['feature_extractor']
            self.trained = True
            print('Classifier loaded successfully (accuracy={})'.format(cache['accuracy']))
        else:
            print('Creating classifier')
            self._cls = svm.LinearSVC()
            self.trained = False

    def _download_dataset(self):
        print('Downloading project dataset')
        print('-' * 24)

        temp_folder = 'temp'
        files_dict = {1: 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip',
                      0: 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'}

        dataset_features = []
        dataset_labels = []
        for label, file_path in files_dict.items():
            file_name = os.path.basename(file_path)
            print('Downloading {}'.format(file_name))
            curl = pycurl.Curl()
            curl.setopt(pycurl.URL, file_path)
            with open(file_name, 'wb') as f:
                curl.setopt(pycurl.WRITEDATA, f)
                curl.setopt(pycurl.NOPROGRESS, False)
                curl.perform()
                curl.close()

            print('Unzipping files')
            with zipfile.ZipFile(file_name) as zip_file:
                progress = tqdm.tqdm(zip_file.infolist())
                for zipped in progress:
                    zip_file.extract(zipped, os.path.join(temp_folder))

            print('Loading images')
            # Load images into collections by folders
            images = {}
            for root, dirs, files in os.walk(temp_folder):
                # Skip MAC hidden folder
                if '_MACOSX' in root:
                    continue
                for img_path in files:
                    if not img_path.endswith('png'):
                        continue

                    # Load with cv2 image and convert in order to make
                    # sure scaling is between 0 to 255 and color space is RBG.
                    img = cv2.imread(os.path.join(root, img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if images.get(root) is None:
                        images[root] = []
                    images[root].append(img)

            print('Randomize samples')

            for key in images.keys():
                shuffle(images[key])

            while len(images):
                key = random.choice(list(images.keys()))
                if len(images[key]) == 0:
                    images.pop(key)
                else:
                    dataset_features.append(images[key].pop())
                    dataset_labels.append(label)

            print('Cleaning up')
            os.remove(file_name)
            shutil.rmtree(temp_folder)

        dataset_features, dataset_labels = shuffle(dataset_features, dataset_labels)
        self._dataset = {'features': np.array(dataset_features), 'labels': np.array(dataset_labels)}

        print('Saving dataset')
        with open(self._dataset_pickle_path, 'wb') as f:
            pickle.dump(self._dataset, f)

    def train(self, feature_extractor: FeatureExtractor, cache_detector: bool=True):
        if os.path.exists(self._dataset_pickle_path):
            print('Loading dataset')
            with open(self._dataset_pickle_path, 'rb') as f:
                self._dataset = pickle.load(f)
        else:
            print('Dataset not found. Starting Download.')
            self._download_dataset()
            print('Dataset download completed.')

        self._feature_extractor = feature_extractor
        print('Extracting features')
        features = feature_extractor(self._dataset['features'])
        labels = self._dataset['labels']

        splitter = StratifiedShuffleSplit(test_size=0.2)
        print('Training classifier')
        accuracy = 0
        for train_idx, test_idx in splitter.split(features, labels):
            X_train, y_train = features[train_idx], labels[train_idx]
            X_test, y_test = features[test_idx], labels[test_idx]

            self._cls.fit(X_train, y_train)
            accuracy = self._cls.score(X_test, y_test)
            print('Accuracy={}'.format(accuracy))

        # Clear dataset from memory
        del self._dataset

        if cache_detector:
            print('Saving detector to {}'.format(self._cached_pickle_path))
            cache = {'cls': self._cls, 'feature_extractor': self._feature_extractor, 'accuracy': accuracy}
            with open(self._cached_pickle_path, 'wb') as f:
                pickle.dump(cache, f)

    def predict(self, X):
        return self._cls.predict(X)

    def get_feature_extractor(self):
        return self._feature_extractor


class DetectionPipeline:
    """
    This class handles & creates the
    detection pipeline
    """
    def __init__(self, detector: VehicleDetector):
        self._first_run = False
        self._detector = detector
        self._feature_extractor = detector.get_feature_extractor()
        self._window_list = None

    def __call__(self, img: np.ndarray):
        # On first run, calculate pipeline properties
        if not self._first_run:
            self._calc_pipeline_properties(img.shape, window_overlap=(0.5, 0.5), window_size_range=(16, 256, 8))
            self._first_run = True

        roi = self._get_region_of_interest(img)
        # Calculate HOG for all the region of interest because its faster then
        # doing this calculation on each detected patch.
        # TODO: hog without feature extraction to roi.
        # hog_roi = self._feature_extractor(roi, hog_features=True, spatial_features=False, color_features=False)
        detected = self._search_windows(img)
        heatmap = self._create_heatmap(detected)
        detected = self._detect_from_heatmap(heatmap, threshold=5)
        return self._draw_boxes(img, detected)

    def _calc_pipeline_properties(self, img_shape, window_overlap, window_size_range):
        self._img_shape = img_shape[0:2]

        # Calc image limits for the region of interest
        self._start_stop_x = [0, img_shape[1]]
        self._start_stop_y = [img_shape[0]//2, img_shape[0]]

        # Calculate the sliding windows map for this region of interest
        avg_size = np.average(window_size_range)
        # Initialize a list to append window positions to
        window_list = []

        span_x = self._start_stop_x[1] - self._start_stop_x[0]
        span_y = self._start_stop_y[1] - self._start_stop_y[0]

        for window_size in range(*window_size_range):
            size = (window_size, window_size)

            # Compute the number of pixels per step in x/y
            step_x = size[0] * (1 - window_overlap[0])
            step_y = size[1] * (1 - window_overlap[1])

            # Compute the number of windows in x/y
            windows_x = int((span_x - size[0]) // step_x + 1)
            windows_y = int((span_y - size[1]) // step_y + 1)

            n_windows = windows_x * windows_y
            # Loop through finding x and y window positions
            for window in range(n_windows):
                # Calculate each window position
                pos_x = np.int(self._start_stop_x[0] + (window % windows_x) * step_x)
                pos_y = np.int(self._start_stop_y[0] + (window // windows_x) * step_y)
                top_left = (pos_x, pos_y)
                bottom_right = (pos_x + size[0], pos_y + size[1])
                # Append window position to list
                window_list.append((top_left, bottom_right))

        self._window_list = window_list

    def _get_region_of_interest(self, img):
        start_x, stop_x = self._start_stop_x
        start_y, stop_y = self._start_stop_y

        return img[start_y:stop_y, start_x:stop_x, :]

    def _search_windows(self, img):

        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in self._window_list:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self._feature_extractor(test_img)
            # 6) Predict using your classifier
            prediction = self._detector.predict(features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def _draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def _create_heatmap(self, detections):
        heatmap = np.zeros(self._img_shape)

        for detection in detections:
            heatmap[detection[0][1]:detection[1][1], detection[0][0]:detection[1][0]] += 1

        return heatmap

    def _detect_from_heatmap(self, heatmap, threshold):
        # Apply threshold
        heatmap[heatmap <= threshold] = 0

        labels = label(heatmap)
        car_boxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            car_boxes.append(bbox)

        return car_boxes

def main():
    input_path = 'test_images'
    output_path = 'output_images'

    print('Vehicle detection script is starting...')

    detector = VehicleDetector(dataset_pickle_path='dataset.p', cached_pickle_path='model.p', load_from_cache=True)
    if not detector.trained:
        feature_extractor = FeatureExtractor(color_space='HLS',
                                             hog_channels='ALL', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2,
                                             spatial_size=(16, 16),
                                             color_hist_bins=32, color_hist_bins_range=(0, 256))
        detector.train(feature_extractor)

    if os.path.isdir(input_path):
        files = os.listdir(input_path)
    else:
        files = [input_path]

    for file in files:
        if os.path.isdir(input_path):
            file_path = os.path.join(input_path, file)
        else:
            file_path = input_path

        if not os.path.isfile(file_path):
            continue

        suffix = file.split('.')[1]
        if suffix == 'jpg':
            # Image processing pipeline
            pipeline = DetectionPipeline(detector)
            img = mpimg.imread(file_path)
            dst = pipeline(img)
            print(file)
            mpimg.imsave(os.path.join(output_path, file), dst)
        elif suffix == 'mp4':
            # Video processing pipeline
            pipeline = DetectionPipeline(detector)
            clip = VideoFileClip(file_path)
            dst = clip.fl_image(pipeline)
            dst.write_videofile(os.path.join(output_path, file), audio=False)

    print('Done. Processed files can be found at {}'.format(output_path))


if __name__ == '__main__':
    main()

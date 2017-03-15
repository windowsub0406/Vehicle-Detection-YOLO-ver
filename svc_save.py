import numpy as np
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extract import extract_features

# Data Exploration - GTI vehicle image dataset
car_images = glob.glob('vehicles/**/*.png', recursive=True)
notcar_images = glob.glob('non-vehicles/**/*.png', recursive=True)

cars = []
notcars = []

for car in car_images:
    cars.append(cv2.imread(car))
for ncar in notcar_images:
    notcars.append(cv2.imread(ncar))

# shuffle images
cars = shuffle(cars)
notcars = shuffle(notcars)

car_image_count = len (cars)
notcar_image_count = len (notcars)

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

car_features = extract_features(cars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                        cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
notcar_features = extract_features(notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                        cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

svc = LinearSVC()
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# save it into a file named svc_save.p
with open("svc_save.p", "wb") as f:
    pickle.dump((svc, X_scaler), f)
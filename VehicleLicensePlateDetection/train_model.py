import numpy as np
from matplotlib import pyplot as plt
import cv2

import os
import glob
from lxml import etree
from tensorflow import keras
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 400
dir_path = os.path.dirname(os.path.realpath(__file__))

img_dir = "input/car-plate-detection/images"  # Enter Directory of all images
data_path = os.path.join(dir_path, img_dir, '*g')
files = glob.glob(data_path)
files.sort()  # We sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes
X = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    X.append(np.array(img))


def resizeannotation(f):
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text) / (width / IMAGE_SIZE)
        ymin = int(dim.xpath("ymin")[0].text) / (height / IMAGE_SIZE)
        xmax = int(dim.xpath("xmax")[0].text) / (width / IMAGE_SIZE)
        ymax = int(dim.xpath("ymax")[0].text) / (height / IMAGE_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]


print(dir_path)
path = os.path.join(dir_path, 'input/car-plate-detection/annotations')
path1 = os.path.join(dir_path, 'input/car-plate-detection/annotations/')
text_files = [path1 + f for f in sorted(os.listdir(path))]
y = []
for i in text_files:
    y.append(resizeannotation(i))

resizeannotation(os.path.join(path1, "Cars147.xml"))

print(y[0])
print(np.array(X).shape)
print(np.array(y).shape)

# plt.figure(figsize=(10, 20))
# for i in range(0, 17):
#     plt.subplot(10, 5, i + 1)
#     plt.axis('off')
#     plt.imshow(X[i])

# Example with the first image of the dataset
# image = cv2.rectangle(X[0], (y[0][0], y[0][1]), (y[0][2], y[0][3]), (0, 0, 255))
# plt.imshow(image)
# plt.show()

# Example with the second image of the dataset
# image = cv2.rectangle(X[1], (y[1][0], y[1][1]), (y[1][2], y[1][3]), (0, 0, 255))
# plt.imshow(image)
# plt.show()

# Transforming in array
X = np.array(X)
y = np.array(y)

# Renormalisation
X = X / 255
y = y / 255

# Convolutionnal Neural Network
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

# Create the model
from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.applications.vgg16 import VGG16

model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=8, verbose=1)

model.save(os.path.join(dir_path, "mode_size_large"), overwrite=True)

# Test
# model = keras.models.load_model(os.path.join(dir_path, "model"))
scores = model.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1] * 100))


# def plot_scores(train):
#     accuracy = train.history['accuracy']
#     val_accuracy = train.history['val_accuracy']
#     epochs = range(len(accuracy))
#     plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
#     plt.plot(epochs, val_accuracy, 'r', label='Score validation')
#     plt.title('Scores')
#     plt.legend()
#     plt.show()
#
#
# plot_scores(train)

# DETECTION
test_loss, test_accuracy = model.evaluate(X_test, y_test, steps=int(100))

print("Test results \n Loss:", test_loss, '\n Accuracy', test_accuracy)

# y_cnn = model.predict(X_test)
# plt.figure(figsize=(20, 40))
# for i in range(0, 43):
#     plt.subplot(10, 5, i + 1)
#     plt.axis('off')
#     ny = y_cnn[i] * 255
#     image = cv2.rectangle(X_test[i], (int(ny[0]), int(ny[1])), (int(ny[2]), int(ny[3])), (0, 255, 0))
#     plt.imshow(image)

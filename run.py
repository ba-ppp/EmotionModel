from handle_csv import show_chart
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# folder

# get the image labels
train_emotion = pd.read_csv('./input/labels.csv')
train_path = './input/train/'

# print(train_emotion.head()) # check csv

# show_chart(train_emotion) # Take a look at distribute

train_labels = train_emotion['type']

# One hot code the labels
one_hot = pd.get_dummies(train_labels, sparse = True)
one_hot_labels = np.asarray(one_hot)

# add the actual path name of the pics to the data set
train_emotion['image_path'] = train_emotion.apply( lambda x: (train_path + x["id"]), axis=1)
train_emotion.head()

# Convert the images to arrays which is used for the model and resize of 299 x 299
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in train_emotion['image_path'].values.tolist()]).astype('float32')
# print(train_data)

# Split the data into train and validation. The stratify parm will insure  train and validation  
x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2, stratify=np.array(train_labels), random_state=100)

# print('x_train shape = ', x_train.shape)
# print('x_validation shape = ', x_validation.shape)

# data = y_train.value_counts().sort_index().to_frame()   # this creates the data frame with train numbers
# data.columns = ['train']   # give the column a name
# data['validation'] = y_validation.value_counts().sort_index().to_frame()   # add the validation numbers
# new_plot = data[['train','validation']].sort_values(['train']+['validation'], ascending=False)   # sort the data
# new_plot.plot(kind='bar', stacked=True)
# plt.show()

y_train = pd.get_dummies(y_train.reset_index(drop=True)).to_numpy()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).to_numpy()

# train genarator
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   # zoom_range = 0.3, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=10)

# # validate genarator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=10, seed=10)

# # inceptionV3 model
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))

# # pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

# # train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                      steps_per_epoch = 175,
                      validation_data = val_generator,
                      validation_steps = 44,
                      epochs = 2,
                      verbose = 2)
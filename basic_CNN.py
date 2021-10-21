from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
import os
import tensorflow as tf #tf 2.0.0
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image, ImageEnhance
import math
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D, Input, BatchNormalization, Activation, SpatialDropout2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import matplotlib as plt
import keras as K
from keras.layers import Input, Conv2D, Dense
from keras.layers import BatchNormalization, SpatialDropout2D
from keras.layers import Activation, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import plot_model

Batch_size = 32
img_h = 256
img_w = 256
num_classes=5

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(img_h, img_w, 3)))

# 1st conv block
model.add(Conv2D(25, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# 2nd conv block
model.add(Conv2D(50, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
# 3rd conv block
model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
#4th
model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())

# ANN block
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.25))
# output layer
model.add(Dense(units=5, activation='softmax'))

# # compile model
# model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# # fit on data for 30 epochs
# model.fit_generator(train, epochs=30, validation_data=val)

train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, 
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,)
#train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

dataset_dir = 'D:/Pragya/bosch/Test_data/Test_data/'



classes = ['airplane', # 0
            'animal', # 1
            'car', # 2
            'human', # 3
            'truck', # 4
           ]

# Training
SEED = 123
tf.random.set_seed(SEED) 

training_dir = os.path.join(dataset_dir, 'Train')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               target_size=(img_h, img_w),
                                               batch_size=Batch_size,
                                               classes=classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
valid_dir = os.path.join(dataset_dir, 'Val')
valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                           target_size=(img_h, img_w),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)
# # Test
test_dir = os.path.join(dataset_dir, 'Test_sorted')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             target_size=(img_h, img_w),
                                             batch_size=10, 
                                             classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED
                                             )

CLASS_NAMES = np.array(['airplane',
            'animal', 
            'car', 
            'human', 
            'truck', ], dtype='<U10')

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.basic.hdf5', save_best_only=True, monitor='val_loss', mode='min')   
callbacks = [earlyStopping, mcp_save,lrr]

Epochnum = 100
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

transfer_learning_history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=Epochnum,
                   callbacks=callbacks,
                  
                  
                    
)


model.evaluate(test_gen, steps=STEP_SIZE_TEST,verbose=1)

model.save("Basic")

import matplotlib.pyplot as plt

acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']

loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']

epochs_range = range(earlyStopping.stopped_epoch+1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
print (epochs_range)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
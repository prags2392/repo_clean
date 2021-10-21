# -*- coding: utf-8 -*-
"""Boschtry.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X2wTj8u2PSQlhGH4Ejymv4Sp10utCu-h
"""



import os
import tensorflow as tf #tf 2.0.0
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image, ImageEnhance

#add brightness variations in the dataset
def change_brightness(dir):
  print ("Processing director "+dir)
  for filename in os.listdir(dir):
    if (filename.startswith('0') and filename.endswith('.jpg')):
      im = Image.open(os.path.join(dir,filename))
      enhancer = ImageEnhance.Brightness(im)

      #factor = 1 #gives original image
      #im_output = enhancer.enhance(factor)
      #im_output.save('original-image.png')

      factor = 0.3 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_3_'+filename))

      factor = 0.5 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_5_'+filename))

      factor = 0.7 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_7_'+filename))

      factor = 0.9 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_9_'+filename))

      factor = 1.2 #brightens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_12_'+filename))

      factor = 1.4 #brightens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_14_'+filename))

# dir = '/content/drive/MyDrive/Bosch/Test_data/Train/'
# change_brightness(os.path.join(dir,'airplane/'))
# change_brightness(os.path.join(dir,'animal/'))
# change_brightness(os.path.join(dir,'car/'))
# change_brightness(os.path.join(dir,'human/'))
# change_brightness(os.path.join(dir,'truck/'))

"""# New Section"""

#train_data_gen = ImageDataGenerator(rescale=1./255,zoom_range=[0.7,1.3],rotation_range=30, horizontal_flip=True)
train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
valid_data_gen = ImageDataGenerator(rescale=1./255)

# valid_data_gen = ImageDataGenerator(rotation_range=30,
#                                     width_shift_range=0.2,
#                                     height_shift_range=0.2,
#                                     zoom_range=0.3,
#                                     horizontal_flip=True,
#                                     vertical_flip=True,
#                                     fill_mode='constant',
#                                     cval=0)

# test_data_gen = ImageDataGenerator(rescale=1./96)

dataset_dir = 'D:/Pragya/bosch/Test_data/Test_data/'

Batch_size = 8
img_h = 96
img_w = 66
num_classes=5

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
                                               target_size=(96, 96),
                                               batch_size=Batch_size,
                                               classes=classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors

# Validation
valid_dir = os.path.join(dataset_dir, 'Val')
valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                           target_size=(96, 96),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)
# # Test
# test_dir = os.path.join(dataset_dir, 'Test')
# test_gen = test_data_gen.flow_from_directory(test_dir,
#                                              target_size=(96, 96),
#                                              batch_size=10, 
#                                              shuffle=False,
#                                              seed=SEED,
#                                              class_mode=None,
#                                              )

CLASS_NAMES = np.array(['airplane',
            'animal', 
            'car', 
            'human', 
            'truck', ], dtype='<U10')

import matplotlib.pyplot as plt

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(25,20))
  for n in range(8):
      ax = plt.subplot(1,8,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
      
# image_batch, label_batch = next(train_gen)
# show_batch(image_batch, label_batch)

DenseNet_model = tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(img_h, img_w, 3), pooling=max)

from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# # The last 15 layers fine tune
# for layer in ResNet_model.layers[:-15]:
#     layer.trainable = False
#layer_name = "conv4_block6_out"

#x = ResNet_model.get_layer(layer_name).output
x = DenseNet_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.4)(x)
# x = Dense(units=128, activation='relu')(x)
# x = Dropout(0.4)(x)
x = Dense(units=64, activation='relu')(x)
x = Dropout(0.4)(x)
output  = Dense(units=5, activation='softmax')(x)
model = Model(DenseNet_model.input, output)


model.summary()

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)


callbacks = [lrr]

Epochnum = 50
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
transfer_learning_history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=Epochnum,
                  #callbacks=callbacks,
                  
                  
                    
)

import matplotlib.pyplot as plt

acc = transfer_learning_history.history['accuracy']
val_acc = transfer_learning_history.history['val_accuracy']

loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']

epochs_range = range(Epochnum)

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

model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

model.save("DeepNetModel")
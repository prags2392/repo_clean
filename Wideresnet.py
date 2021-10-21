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

Batch_size = 8
img_h = 128
img_w = 128
num_classes=5

def resnet_input_block(input_layer, 
                       num_filters):
    
    x = Conv2D(filters=num_filters, 
               kernel_size=3, 
               padding='same')(input_layer)
    
    x = BatchNormalization(axis=-1)(x)
    
    x_out = Activation('relu')(x)
    
    return x_out


def resnet_block(x_in, 
                 num_filters, 
                 num_miniblocks, 
                 downsampling,
                 dropout_rate=0.0):
    '''
    ** Input Parameters **
    
    x_in : input layer tensor (batchsize, width, height, channels)
    
    num_filters : total number of feature maps including
                  the widening factor multiplier
                 
    num_miniblocks : number of times we repeat the miniblock
                     (a.k.a. "groupsize" in the WRN paper)
                    
    downsampling : factor by which we reduce the output size, 
                   i.e. downsampling of 2 takes 16x16 --> 8x8;
                   downsampling here is applied by adjusting 
                   the stride of the first convolutional layers
                   rather than via pooling layers. 
                  
    ** Returns **
    
    x_out : output layer for the next network block
    
    '''
    
    # residual path
    x_res = Conv2D(filters=num_filters, 
                   kernel_size=1, 
                   padding='same',
                   strides=downsampling)(x_in)
    
    # main path mini-block
    x_main = Conv2D(filters=num_filters, 
                    kernel_size=3, 
                    padding='same',
                    strides=downsampling)(x_in)    
    
    x_main = BatchNormalization(axis=-1)(x_main)
    
    x_main = Activation('relu')(x_main)
    
    x_main = SpatialDropout2D(rate=dropout_rate)(x_main)
    
    x_main = Conv2D(filters=num_filters, 
                    kernel_size=3, 
                    padding='same')(x_main)
    
    # merge
    x_out = K.layers.Add()([x_main, x_res])
    
    
    # *** additional miniblocks ***
    for block in range(num_miniblocks-1):
        
        # main path mini-block
        x_main = BatchNormalization(axis=-1)(x_out)
        
        x_main = Activation('relu')(x_main)

        x_main = Conv2D(filters=num_filters, 
                        kernel_size=3, 
                        padding='same')(x_main)
        
        x_main = SpatialDropout2D(rate=dropout_rate)(x_main)
        
        x_main = BatchNormalization(axis=-1)(x_main)
        
        x_main = Activation('relu')(x_main)
        
        x_main = Conv2D(filters=num_filters, 
                        kernel_size=3, 
                        padding='same')(x_main)
        
        # merge
        x_out = K.layers.Add()([x_main, x_out])
    # ***
    
    
    # capping the resnet block 
    x = BatchNormalization(axis=-1)(x_out)

    x_out = Activation('relu')(x)
      
    return x_out


def resnet_output_block(x_in):
    
    # auto-adjust pooling based on input shape
    #dim1_ = x_in.shape[1].value
    dim1_ = x_in.shape.as_list()[1]
    #dim2_ = x_in.shape[2].value
    dim2_ = x_in.shape.as_list()[2]
    assert dim1_ == dim2_, 'Input layer dimensions must be square.'

    # this generates a single average value for each feature map
    x = K.layers.AveragePooling2D(pool_size=dim1_)(x_in)
    
    # flatten and apply a fully-connected layer 
    x = K.layers.Flatten()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.45)(x)
    # obtain probabilities for each class
    class_probas = Dense(5, activation='softmax')(x)
    
    return class_probas


def build_WRN(num_convs=40, k_factor=2, drop_rate=0.0):
    """
    Builds a wide residual network (WRN) of type
    WRN-N-K where N is the total number of convolutions
    and K is the widening factor. 
    
    ** Input Parameters **
    
    num_convs: Total number of convolutions. Must be an 
               integer value derivable from (i * 2 * 3) + 4, 
               where i is an integer (e.g. 10, 16, 22, 28, 
               34, and 40 are acceptable values)
               
    k_factor: Widening factor, multiplies the number of channels
              in each resnet block convolution. Must be an 
              integer value. 
              
    drop_rate: Fraction of features dropped during SpatialDropout2D
               between conv layers in each residual block. 
              
    ** Returns **
    
    model: WRN model object built with Keras' functional API. 
    
    """
    assert type(num_convs) is int, \
    "num_convs is not an integer: %r" % num_convs
        
    assert type(k_factor) is int, \
    "k_factor is not an integer: %r" % k_factor
        
    assert num_convs in [10, 16, 22, 28, 34, 40], \
    "num_convs must be one of [10, 16, 22, 28, 34, 40]"
    
    num_miniblocks = int(math.ceil((num_convs - 4)/6))

    input_layer = Input(shape=(img_h, img_w, 3))

    # input block ('conv1' group in WRN paper)
    x = resnet_input_block(input_layer, num_filters=16)

    # ('conv2' group in WRN paper)
    x = resnet_block(x_in=x, 
                     num_filters=16*k_factor, 
                     num_miniblocks=num_miniblocks, 
                     downsampling=2,
                     dropout_rate=drop_rate)

    # ('conv3' group in WRN paper)
    x = resnet_block(x_in=x, 
                     num_filters=32*k_factor, 
                     num_miniblocks=num_miniblocks, 
                     downsampling=2,
                     dropout_rate=drop_rate)

    # ('conv4' group in WRN paper)
    x = resnet_block(x_in=x, 
                     num_filters=64*k_factor, 
                     num_miniblocks=num_miniblocks, 
                     downsampling=2,
                     dropout_rate=drop_rate)

    # output block 
    probas = resnet_output_block(x_in=x)
    
    # assemble
    model = K.Model(inputs=input_layer, outputs=probas)
    
    return model

K.backend.clear_session()
model_WRN = build_WRN(40, 2, drop_rate=0.45)
model_WRN.summary()


train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, 
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,)
# train_data_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, 
#         rotation_range=30,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.3,
#         zoom_range=[1,1.5],
#         vertical_flip=True,)
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
model_WRN.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.wideresnetv4_imagesize128Conv40_batch8.hdf5', save_best_only=True, monitor='val_loss', mode='min')   
callbacks = [earlyStopping, mcp_save, lrr]

Epochnum = 100
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

transfer_learning_history = model_WRN.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=Epochnum,
                   callbacks=callbacks,
                  
                  
                    
)


model_WRN.evaluate(test_gen, steps=STEP_SIZE_TEST,verbose=1)

model_WRN.save("WideResNEt_128_conv40_batch8_dropout0.45")

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

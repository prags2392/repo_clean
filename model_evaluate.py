import os
import tensorflow as tf #tf 2.0.0
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image, ImageEnhance
import shutil

dataset_dir = 'D:/Pragya/bosch/Test_data/Test_data/'
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

Batch_size = 8
img_h = 96
img_w = 96
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

# # Test
test_dir = os.path.join(dataset_dir, 'Test_sorted')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             target_size=(96, 96),
                                             batch_size=1, 
                                             classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED
                                             )


# Validation
valid_dir = os.path.join(dataset_dir, 'Val')
valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                           target_size=(96, 96),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)

#ResNet_model = tf.keras.applications.InceptionResNetV2(weights=None, include_top=False, input_shape=(img_h, img_w, 3))
ResNet_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(img_h, img_w, 3))

from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


# # The last 15 layers fine tune
# for layer in ResNet_model.layers[:-15]:
#     layer.trainable = False
#layer_name = "conv4_block6_out"

#x = ResNet_model.get_layer(layer_name).output
x = ResNet_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=256, activation='relu')(x)
x = Dropout(0.4)(x)
# x = Dense(units=128, activation='relu')(x)
# x = Dropout(0.4)(x)
x = Dense(units=64, activation='relu')(x)
x = Dropout(0.4)(x)
output  = Dense(units=5, activation='softmax')(x)
model = Model(ResNet_model.input, output)

#model.load_weights('.inceptionregmdl_wts.hdf5')
model.load_weights('.l1l2regmdl_wtsv2.hdf5')
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
# model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
# model.evaluate(test_gen, steps=STEP_SIZE_TEST,verbose=1)

# predict = model.predict_generator(test_gen, steps=STEP_SIZE_TEST)
# ans = np.argmax(predict, axis=1)
# print (ans)

from keras.preprocessing import image
import matplotlib.pyplot as plt

valpath = "D:\\Pragya\\bosch\\Test_data\\Test_data\\Val\\airplane"
despath = ""

def prepare(img_path):
    img = image.load_img(img_path, target_size=(96,96))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

def result(path):    
	result = model.predict([prepare(path)])
	d=image.load_img(path)
	plt.imshow(d)
	x=np.argmax(result,axis=1)
	return (classes[int(x)])

for files in os.listdir(valpath):
	imgp = os.path.join(valpath,files)
	print ("For file: "+files)
	res = result(imgp)
	print ("Result is: "+ res)
	p1 = os.path.join('D:\\Pragya\\bosch\\wrong_val\\airplane',res)
	if(res != 'animal'):
		shutil.copy(imgp, os.path.join(p1,files))
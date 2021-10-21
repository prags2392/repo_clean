import os
import tensorflow as tf #tf 2.0.0
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image, ImageEnhance
import shutil

modelname	= 'Resnet152V2_Img128_onlyHorizFlip_batch16_TrainVal'
model = tf.keras.models.load_model(modelname)
model.summary()

dataset_dir = 'D:/Pragya/bosch/Test_data/Test_data/'
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)


SEED = 123
tf.random.set_seed(SEED) 

classes = ['airplane', # 0
            'animal', # 1
            'car', # 2
            'human', # 3
            'truck', # 4
           ]


Batch_size = 8
img_h = 128
img_w = 128
num_classes=5
# # Test
test_dir = os.path.join(dataset_dir, 'Test_sorted')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             target_size=(img_h,img_w),
                                             batch_size=1, 
                                             classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED
                                             )


# Validation
valid_dir = os.path.join(dataset_dir, 'Val')
valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                           target_size=(img_h,img_w),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)

STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
print ("Validation Accuracy")
model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)

STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
print ("Test Accuracy")
model.evaluate(test_gen, steps=STEP_SIZE_TEST,verbose=1)

from keras.preprocessing import image
import matplotlib.pyplot as plt

def prepare(img_path):
    img = image.load_img(img_path, target_size=(img_h,img_w))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

def result(path):    
	result = model.predict([prepare(path)])
	d=image.load_img(path)
	plt.imshow(d)
	x=np.argmax(result,axis=1)
	return (classes[int(x)])

# for string in classes:
# 	print (string)
# 	valpath = os.path.join("D:\\Pragya\\bosch\\Test_data\\Test_data\\Test_sorted",string)
# 	dpath = "D:\\Pragya\\bosch\\wrong_test_with_widenet_256"
# 	if not (os.path.isdir(dpath)):
# 		os.mkdir(dpath)
# 	despath = os.path.join(dpath,string)
# 	if not (os.path.isdir(despath)):
# 		os.mkdir(despath)

# 	for files in os.listdir(valpath):
# 		imgp = os.path.join(valpath,files)
# 		#print ("For file: "+files)
# 		res = result(imgp)
# 		#print ("Result is: "+ res)
# 		p1 = os.path.join(despath,res)
# 		if not(os.path.isdir(p1)):
# 			os.mkdir(p1)
# 		if(res != string):
# 			shutil.copy(imgp, os.path.join(p1,files))
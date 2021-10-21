import cv2
from PIL import Image, ImageEnhance
import os

#add brightness variations in the dataset
def change_brightness(dir):
  print ("Processing directory "+dir)
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

      factor = 0.6 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_6_'+filename))

      factor = 0.7 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_7_'+filename))

      factor = 0.8 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_8_'+filename))

      factor = 0.9 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_9_'+filename))

      factor = 1.1 #darkens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_11_'+filename))

      factor = 1.2 #brightens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_12_'+filename))

      factor = 1.3 #brightens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_13_'+filename))

      factor = 1.4 #brightens the image
      im_output = enhancer.enhance(factor)
      im_output.save(os.path.join(dir,'darkened-image_14_'+filename))

dir = 'D:/Pragya/bosch/Test_data/Test_data/TrainVal'
change_brightness(os.path.join(dir,'airplane/'))
change_brightness(os.path.join(dir,'animal/'))
change_brightness(os.path.join(dir,'car/'))
change_brightness(os.path.join(dir,'human/'))
change_brightness(os.path.join(dir,'truck/'))
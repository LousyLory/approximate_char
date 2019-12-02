import numpy as np
import os
import cv2
import sys
from glob import glob
 
 
images_dir_path = sys.argv[1]
save_to_dir_path = sys.argv[2]
 
image_filenames = glob(os.path.join(images_dir_path, "*.tiff"))
print(image_filenames)
 
if not os.path.exists(save_to_dir_path):
    os.mkdir(os.path.join(images_dir_path, "cropped_images"))
 
for i, image_file in enumerate(image_filenames):
   
    image_original_name = image_file
    file_path_pieces = image_original_name.split(os.sep)
    file_stripped_name = file_path_pieces[-1].split('.')[0]
    print(file_stripped_name)
    current_image = cv2.imread(image_file)
    (width, height, channels) = current_image.shape
 
    image_x_max = width
    image_y_max = height
 
    image_crop_x = 1280
    x_0 = 0
    image_crop_y = 720
    y_0 = 0
    image_crop_step = 200
 
    while y_0 + image_crop_y < image_y_max:
        while x_0 + image_crop_x < image_x_max:
            save_to_image = os.path.join(save_to_dir_path, "cropped_images", ("cropped_image_%s_%d_%d.tiff" % (file_stripped_name, x_0 , y_0)))
            cropped = current_image[y_0:y_0+image_crop_y, x_0:x_0+image_crop_x]
            cv2.imwrite(save_to_image, cropped)
 
            x_0 += image_crop_step
 
        x_0 = 0
        y_0 += image_crop_step
# NECESSARY PYTHON LIBRARIES
import os
import cv2
from glob import glob

# THIS FUNCTION RETURNS TINY IMAGES ARRAY, LABELS ARRAY
def get_tiny_images(path, size, num_img_per_cat, categories):

    tiny_images = []
    labels = []

    for category in categories:

        image_paths = glob(os.path.join(path, category, '*.jpg'))
        for i in range(num_img_per_cat):

            image = cv2.imread(image_paths[i])
            image = cv2.resize(image, size).flatten()
            tiny_images.append(image)
            labels.append(category)

    return tiny_images, labels

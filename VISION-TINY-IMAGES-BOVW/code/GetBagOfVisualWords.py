# NECESSARY PYTHON LIBRARIES
import numpy as np
import cv2
import os
from glob import glob
from scipy.cluster.vq import kmeans, vq

# THIS FUNCTION RETURNS DICTIONARY THAT HOLS IMAGES PATHS AND THEIR CATEGORIES
def create_image_dict(path, num_img_per_cat, categories):

    images = {}

    for category in categories:
        image_paths = glob(os.path.join(path, category, '*.jpg'))

        for i in range(num_img_per_cat):
            images[image_paths[i]] = category

    return images

# THIS FUNCTION RETURNS DESCRIPTOR LIST
def sift_of_images(image_dict):

    descriptor_list = []
    sift = cv2.SIFT_create()

    for image_path in image_dict.keys():
        image = cv2.imread(image_path)
        keypoint, descriptor = sift.detectAndCompute(image, None)
        descriptor_list.append(descriptor)

    return descriptor_list

# THIS FUNCTION RETURNS VOCABULARY CREATED WITH K-MEANS CLUSTERING
def k_means_clustering(descriptors):

    vocabulary, variance = kmeans(descriptors, 6, 1)

    return vocabulary, variance

# THIS FUNCTION RETURNS HISTOGRAM OF IMAGES (FEATURES OF IMAGES)
def histogram_of_images(image_dict, vocabulary, des_list):

    image_feats = np.zeros((len(list(image_dict.keys())), 6), "float32")

    for i in range(len(list(image_dict.keys()))):
        words, dist = vq(list(des_list[i]), vocabulary)

        for word in words:
            image_feats[i][word] += 1

    return image_feats











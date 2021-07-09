# NECESSARY PYTHON LIBRARIES
import os
import numpy as np
import shutil

# DATASET PATH, CATEGORIES
scene_dataset_path = 'C:/Users/can_a/PycharmProjects/Assignment2/SceneDataset'
categories = ['Bedroom', 'Highway', 'Kitchen', 'LivingRoom', 'Mountain', 'Office']

# CREATING TRAIN TEST SET (0.30 TEST, 0.70 TRAIN)
for i in categories:
    os.makedirs(scene_dataset_path + '/train/' + i)
    os.makedirs(scene_dataset_path + '/test/' + i)
    dataset = scene_dataset_path + '/' + i

    all_images = os.listdir(dataset)
    np.random.shuffle(all_images)

    test_ratio = 0.3
    train_set, test_set = np.split(np.array(all_images), [int(len(all_images) * (1 - test_ratio))])

    train_set = [dataset + '/' + name for name in train_set.tolist()]
    test_set = [dataset + '/' + name for name in test_set.tolist()]

    for name in train_set:
        shutil.copy(name, scene_dataset_path + '/train/' + i)

    for name in test_set:
        shutil.copy(name, scene_dataset_path + '/test/' + i)

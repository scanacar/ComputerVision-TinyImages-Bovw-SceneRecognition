# NECESSARY PYTHON LIBRARIES
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

# IMPORTING FUNCTIONS FROM OTHER CLASSES
from GetTinyImage import get_tiny_images
from GetBagOfVisualWords import create_image_dict, sift_of_images, k_means_clustering, histogram_of_images

DATASET = 'C:/Users/can_a/PycharmProjects/Assignment2/SceneDataset'  # DATASET PATH FROM MY COMPUTER
CATEGORIES = ['Bedroom', 'Highway', 'Kitchen', 'LivingRoom', 'Mountain', 'Office']  # TRAIN-TEST SET CATEGORIES
NUM_IMG_PER_CAT = 60  # NUMBER IMAGE PER CATEGORIES
SIZE = (16, 16)  # TINY IMAGES SIZE


def main():

    train_paths = 'C:/Users/can_a/PycharmProjects/Assignment2/SceneDataset/train'  # TRAIN SET PATH
    test_paths = 'C:/Users/can_a/PycharmProjects/Assignment2/SceneDataset/test'   # TEST SET PATH

    # TINY IMAGE FEATURES AND K-NEAREST NEIGHBORS

    train_features, train_labels = get_tiny_images(train_paths, SIZE, NUM_IMG_PER_CAT, CATEGORIES)     # GETTING TRAIN TINY IMAGES AND TRAIN LABELS (X_TRAIN, Y_TRAIN)
    test_features, test_labels = get_tiny_images(test_paths, SIZE, NUM_IMG_PER_CAT, CATEGORIES)        # GETTING TEST TINY IMAGES AND TEST LABELS (X_TEST, Y_TEST)

    # CONVERTING TINY IMAGES ARRAYS AND LABELS ARRAYS TO NUMPY ARRAY
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # K-NN MODEL CREATING, FITTING, PREDICTING
    model_knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    model_knn.fit(train_features, train_labels)    # PARAMETERS: X_TRAIN, Y_TRAIN
    tiny_knn_pred = model_knn.predict(test_features)   # PARAMETERS: X_TEST
    accuracy_tiny_knn = model_knn.score(test_features, test_labels)  # PARAMETERS: X_TEST, Y_TEST

    # print(list(tiny_knn_pred[300:359]))  # To select true and false predicted images

    # True Predicted: Bedroom(4, 11, 14, 17, 26), Highway(2, 6, 8, 10, 11), Kitchen(4, 5, 7, 34, 35)
    # True Predicted: LivingRoom(22, 36, 43, 47), Mountain(5, 6, 7, 8, 10), Office(2, 3, 6, 8, 9)
    # False Predicted: Office(1-Mountain, 4-Bedroom, 5-Kitchen, 7-Mountain, 11-Highway)

    print("Tiny image features and KNN accuracy: {:.2f}".format(accuracy_tiny_knn))  # PRINTING ACCURACY SCORE

    # PLOTTING CONFUSION MATRIX FOR TINY IMAGES AND K-NN
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.title("Tiny image features and KNN confusion matrix")
    plot_confusion_matrix(model_knn, test_features, test_labels, normalize="true", xticks_rotation="vertical", cmap=plt.cm.Blues, ax=ax)
    plt.show()

    # TINY IMAGE FEATURES AND LINEAR SVM

    # CREATING MINMAXSCALER FOR LINEAR SVM
    mms = MinMaxScaler()

    # SCALING X_TRAIN, X_TEST
    train_mm_features = mms.fit_transform(train_features)
    test_mm_features = mms.fit_transform(test_features)

    # CREATING MODEL (LINEAR SVM)
    svm = LinearSVC(random_state=42, C=0.01)
    svm.fit(train_mm_features, train_labels)  # PARAMETERS: X_TRAIN, Y_TRAIN
    y_pred = svm.predict(test_mm_features)  # PARAMETERS: X_TEST

    accuracy_tiny_svm = accuracy_score(test_labels, y_pred)  # PARAMETERS: Y_TEST, Y_PRED

    print("Tiny image features and Linear SVM accuracy: {:.2f}".format(accuracy_tiny_svm))  # PRINTING ACCURACY SCORE

    # PLOTTING CONFUSION MATRIX FOR TINY IMAGES AND SVM
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.title('Tiny image features and Linear SVM confusion matrix')
    plot_confusion_matrix(svm, test_mm_features, test_labels, normalize="true", xticks_rotation="vertical", cmap=plt.cm.Blues, ax=ax)
    plt.show()

    # BAG OF VISUAL WORDS AND K-NEAREST NEIGHBORS

    # CREATING DICTIONARY THAT HOLDS IMAGE PATHS AND CATEGORIES
    train_dict = create_image_dict(train_paths, NUM_IMG_PER_CAT, CATEGORIES)  # Image path (key), Category (value)
    test_dict = create_image_dict(test_paths, NUM_IMG_PER_CAT, CATEGORIES)  # Image path (key), Category (value)

    # CREATING DESCRIPTOR LIST FROM DICTIONARIES
    train_des_list = sift_of_images(train_dict)
    test_des_list = sift_of_images(test_dict)

    # STACKING TRAIN DESCRIPTOR LIST PARTIALLY
    descriptors = train_des_list[0][1]
    for descriptor in train_des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    descriptors = descriptors.astype(float)  # CONVERTING TYPE TO FLOAT

    vocabulary, variance = k_means_clustering(descriptors)     # K-MEANS CLUSTERING FOR CREATING VOCABULARY

    # CREATING HISTOGRAM FOR TRAIN AND TEST IMAGES
    train_bow = histogram_of_images(train_dict, vocabulary, train_des_list)
    test_bow = histogram_of_images(test_dict, vocabulary, test_des_list)

    # CREATING MODEL K-NN FOR BOW (DICT.VALUES = LABELS)
    model_knn_bow = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    model_knn_bow.fit(train_bow, list(train_dict.values()))        # PARAMETERS: X_TRAIN, Y_TRAIN
    y_pred_knn_bow = model_knn_bow.predict(test_bow)               # PARAMETERS: X_TEST
    accuracy_knn_bow = model_knn_bow.score(test_bow, list(test_dict.values()))   # PARAMETERS: X_TEST, Y_TEST

    print("Bag of Visual Words and KNN accuracy: {:.2f}".format(accuracy_knn_bow))  # PRINTING ACCURACY SCORE

    # PLOTTING CONFUSION MATRIX FOR BOW AND K-NN
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.title("Bag of Visual Words and KNN confusion matrix")
    plot_confusion_matrix(model_knn_bow, test_bow, list(test_dict.values()), normalize="true", xticks_rotation="vertical", cmap=plt.cm.Blues, ax=ax)
    plt.show()

    # BAG OF VISUAL WORDS AND LINEAR SVM

    # CREATING STANDARD SCALER
    ss = StandardScaler()

    # SCALING TRAIN_BOW AND TEST_BOW (X_TRAIN, X_TEST)
    ss_train_bow = StandardScaler().fit_transform(train_bow)
    ss_test_bow = StandardScaler().fit_transform(test_bow)

    # CREATING LINEAR SVM MODEL
    svm_bow = LinearSVC(random_state=42, C=0.01)
    svm_bow.fit(ss_train_bow, list(train_dict.values()))  # PARAMETERS: X_TRAIN, Y_TRAIN
    y_pred_svm_bow = svm_bow.predict(ss_test_bow)   # PARAMETERS: X_TEST

    accuracy_bow_svm = accuracy_score(list(test_dict.values()), y_pred_svm_bow)  # PARAMETERS: Y_TEST, Y_PRED

    print("Bag of Visual Words and Linear SVM accuracy: {:.2f}".format(accuracy_bow_svm))  # PRINTING ACCURACY SCORE

    # PLOTTING CONFUSION MATRIX FOR BOW AND SVM
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.title("Bag of Visual Words and Linear SVM confusion matrix")
    plot_confusion_matrix(svm_bow, ss_test_bow, list(test_dict.values()), normalize="true", xticks_rotation="vertical", cmap=plt.cm.Blues, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()

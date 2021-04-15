import cv2
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time

# This function will sample SIFT descriptors from the training images,
# cluster them with kmeans, and then return the cluster centers.


def build_vocabulary(image_paths, vocab_size):
    ##################################################################################
    # Load images from the training set. To save computation time, you don't         #
    # necessarily need to sample from all images, although it would be better        #
    # to do so. You can randomly sample the descriptors from each image to save      #
    # memory and speed up the clustering. Or you can simply call vl_dsift with       #
    # a large step size here.                                                        #
    #                                                                                #
    # For each loaded image, get some SIFT features. You don't have to get as        #
    # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
    # trying to get a representative sample here.                                    #
    #                                                                                #
    # Once you have tens of thousands of SIFT features from many training            #
    # images, cluster them with kmeans. The resulting centroids are now your         #
    # visual word vocabulary.                                                        #
    ##################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # This function will sample SIFT descriptors from the training images,           #
    # cluster them with kmeans, and then return the cluster centers.                 #
    #                                                                                #
    # Function : dsift()                                                             #
    # SIFT_features is a N x 128 matrix of SIFT features                             #
    # There are step, bin size, and smoothing parameters you can                     #
    # manipulate for dsift(). We recommend debugging with the 'fast'                 #
    # parameter. This approximate version of SIFT is about 20 times faster to        #
    # compute. Also, be sure not to use the default value of step size. It will      #
    # be very slow and you'll see relatively little performance gain from            #
    # extremely dense sampling. You are welcome to use your own SIFT feature.        #
    #                                                                                #
    # Function : kmeans(X, K)                                                        #
    # X is a M x d matrix of sampled SIFT features, where M is the number of         #
    # features sampled. M should be pretty large!                                    #
    # K is the number of clusters desired (vocab_size)                               #
    # centers is a d x K matrix of cluster centroids.                                #
    #                                                                                #
    # NOTE:                                                                          #
    #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
    #        2. kmeans( ? , vocab_size)                                              #
    #                                                                                #
    # ################################################################################
    """
    Input :
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    """

    n = len(image_paths)
    sift_descriptors = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _, descriptors = dsift(img, step=(20, 20), fast=True, float_descriptors=True)
        sift_descriptors.append(descriptors)

    sift_descriptors = np.vstack(sift_descriptors)
    print(sift_descriptors.shape)

    vocab = kmeans(sift_descriptors, vocab_size, initialization="PLUSPLUS")
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    return vocab

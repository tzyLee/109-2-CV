import cv2
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time


def get_bags_of_sifts(image_paths):
    ############################################################################
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    """
    Input :
        image_paths : a list(N) of training images
    Output :
        image_feats : (N, d) feature, each row represent a feature of an image
    """
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vocab_size = vocab.shape[0]
    histograms = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Get sift descriptors
        _, descriptors = dsift(img, step=(5, 5), fast=True, float_descriptors=True)
        # Get nearest centroid of each descriptor
        descriptor_count = descriptors.shape[0]
        dist = distance.cdist(descriptors, vocab)
        argmin = dist.argmin(axis=1)
        # Get histogram of occurrence of vocab
        histogram = np.bincount(argmin, minlength=vocab_size).astype(np.float32)
        histogram /= descriptor_count  # normalize histogram
        histograms.append(histogram)
    image_feats = np.vstack(histograms)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats

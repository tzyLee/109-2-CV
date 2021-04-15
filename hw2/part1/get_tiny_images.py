import cv2
import numpy as np

def get_tiny_images(image_paths):
    #############################################################################
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input :
        image_paths: a list(N) of string where each string is an image
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    n = len(image_paths)
    tiny_images = np.zeros((n, 16*16))
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (16, 16))
        tiny_images[i, :] = resized.ravel()

    # zero mean
    tiny_images -= tiny_images.mean(axis=0)
    # unit length
    tiny_images /= np.linalg.norm(tiny_images, axis=1, keepdims=True)
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images

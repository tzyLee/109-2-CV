import numpy as np
import cv2
import matplotlib.pyplot as plt


def is_max(score, x, y, w, h):
    for i in range(x - 2, x + 3):
        for j in range(y - 2, y + 3):
            if (
                0 <= i < w
                and 0 <= j < h
                and (i, j) != (x, y)
                and score[i, j] >= score[x, y]
            ):
                return False
    return True


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.diff_x = np.array([[1.0, 0.0, -1.0]])
        self.diff_y = self.diff_x.T

    def detect_harris_corners(self, img):
        # Step 1: Smooth the image by Gaussian kernel
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.5)
        cv2.GaussianBlur(img, (3, 3), 1.5, img)

        # Step 2: Calculate Ix, Iy (1st derivative of image along x and y axis)
        # - Function: cv2.filter2D (kernel = [[1.,0.,-1.]] for Ix or [[1.],[0.],[-1.]] for Iy)
        Ix = cv2.filter2D(img, -1, self.diff_x)
        Iy = cv2.filter2D(img, -1, self.diff_y)

        # Step 3: Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        # Step 4: Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.)
        Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1)
        Syy = cv2.GaussianBlur(Iyy, (3, 3), 1)

        # Step 5: Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy

        # Step 6: Compute the response of the detector by det/(trace+1e-12)
        response = det / (trace + 1e-12)

        return response

    def post_processing(self, response):
        # Step 1: Thresholding
        filtered = response > self.threshold
        candidates = np.argwhere(filtered)
        score = response[filtered]

        local_max = []
        pick = local_max.append

        # Step 2: Non-Maximum Suppression
        w, h = response.shape
        for x, y in candidates:
            if is_max(response, x, y, w, h):
                pick([x, y])

        return local_max
import numpy as np
import cv2
import random
from itertools import chain
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max(x.shape[0] for x in imgs)
    w_max = sum(x.shape[1] for x in imgs)

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[: imgs[0].shape[0], : imgs[0].shape[1]] = imgs[0]
    accumulatedH = np.eye(3)
    out = None
    orb = cv2.ORB_create(nfeatures=320)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    rng = np.random.default_rng(seed=82)

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bf.match(des1, des2)
        matches.sort(key=lambda m: m.distance)
        matches = matches[: int(0.92 * len(matches))]
        nMatch = len(matches)

        # Transform List[cv2.KeyPoint] into np.array of [[x, y]]
        kp1 = np.fromiter(
            chain.from_iterable(kp1[match.queryIdx].pt for match in matches),
            dtype=np.float64,
            count=nMatch * 2,
        ).reshape((-1, 2))
        kp2 = np.fromiter(
            chain.from_iterable(kp2[match.trainIdx].pt for match in matches),
            dtype=np.float64,
            count=nMatch * 2,
        ).reshape((-1, 2))
        # kp1 padded with ones
        kp2Full = np.hstack((kp2, np.ones((nMatch, 1), dtype=np.float64)))

        # 2. apply RANSAC to choose best H
        it = 0
        nIter = 100000
        maxNInlier = 0
        maxInlierMask = None
        nInlierThreshold = nMatch * 0.95
        distThreshold = 0.94
        p = 0.96  # Probability of at least one sample is free of outliers

        # Repeat for N iterations, select the largest inlier set
        while it < nIter:
            # Sample s data points to instantiate model
            sampleIndices = rng.choice(nMatch, 4, replace=False)
            sample1 = kp1[sampleIndices]
            sample2 = kp2[sampleIndices]
            H = solve_homography(sample2, sample1)
            H_T = np.transpose(H)

            # Find the inliers (within distance threshold t)
            x2 = kp2Full @ H_T
            x2[:, :2] /= x2[:, 2, np.newaxis] + 1e-16

            dist = np.linalg.norm(x2[:, :2] - kp1, axis=1)

            inlierMask = dist < distThreshold
            nInlier = np.count_nonzero(inlierMask)

            if nInlier > maxNInlier:
                maxNInlier = nInlier
                maxInlierMask = inlierMask

            # If size of inliers >= T, re-estimate the model and terminate
            if maxNInlier >= nInlierThreshold:
                break

            # Adaptively determine the number of sample
            nIter = np.log(1 - p) / np.log(1 - (nInlier / nMatch) ** 4 + 1e-14)
            it += 1

        # The homography is re-estimated using all points in the maximum inlier set
        H = solve_homography(kp2[maxInlierMask], kp1[maxInlierMask])

        # 3. chain the homographies
        accumulatedH = accumulatedH @ H

        # 4. apply warping
        out = warping(im2, dst, accumulatedH, 0, h_max, 0, w_max, direction="b")
    return out


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [
        cv2.imread("../resource/frame{:d}.jpg".format(x))
        for x in range(1, FRAME_NUM + 1)
    ]
    output4 = panorama(imgs)
    cv2.imwrite("output4.png", output4)

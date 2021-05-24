import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import chain
from utils import solve_homography, warping


def RANSAC(img1, img2):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    rng = np.random.default_rng(seed=4146458350)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
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

    it = 0
    nIter = 100000
    maxNInlier = 0
    maxInlierMask = None
    distThreshold = 3
    p = 0.995  # Probability of at least one sample is free of outliers

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

        inlierMask = dist <= distThreshold
        nInlier = np.count_nonzero(inlierMask)

        if nInlier > maxNInlier and nInlier >= 4:
            maxNInlier = nInlier
            maxInlierMask = inlierMask

            # Adaptively determine the number of sample
            nIter = np.log(1 - p) / np.log(1 - (nInlier / nMatch) ** 4 + 1e-16)

        it += 1
    return solve_homography(kp2[maxInlierMask], kp1[maxInlierMask])


def get_canvas(h_max, w_max, channels, dtype=np.float32):
    return np.zeros((h_max, w_max, channels), dtype=dtype)


def get_mask(img, H, h_max, w_max, channels):
    mask = get_canvas(h_max, w_max, channels, dtype=np.float32)
    temp = np.full_like(img, 128, dtype=np.float32)
    mask = warping(temp, mask, H, 0, h_max, 0, w_max, direction="b")
    np.clip(mask, 0, 255, out=mask)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return cv2.distanceTransform(mask, cv2.DIST_L1, 3)


def warp(img, H, h_max, w_max, channels):
    dst = get_canvas(h_max, w_max, channels, dtype=np.float32)
    return warping(img, dst, H, 0, h_max, 0, w_max, direction="b")


imgs = [
    cv2.imread("../resource/frame1.jpg"),
    cv2.imread("../resource/frame2.jpg"),
    cv2.imread("../resource/frame3.jpg"),
]
h_max = max(x.shape[0] for x in imgs)
w_max = sum(x.shape[1] for x in imgs)
channels = imgs[0].shape[2]

dist1 = get_mask(imgs[0], np.eye(3), h_max, w_max, channels)
H1 = RANSAC(imgs[0], imgs[1])
dist2 = get_mask(imgs[1], H1, h_max, w_max, channels)

alpha = dist1 / (dist1 + dist2 + 1e-16)
dst1 = alpha[..., np.newaxis] * warp(imgs[0], np.eye(3), h_max, w_max, channels) + (
    1 - alpha[..., np.newaxis]
) * warp(imgs[1], H1, h_max, w_max, channels)

temp = alpha[..., np.newaxis] * warp(imgs[0] + 1, np.eye(3), h_max, w_max, channels) + (
    1 - alpha[..., np.newaxis]
) * warp(imgs[1] + 1, H1, h_max, w_max, channels)
temp[temp > 0] = 255
temp = temp.astype(np.uint8)
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

dist12 = cv2.distanceTransform(temp, cv2.DIST_L1, 3)
H2 = RANSAC(imgs[1], imgs[2])
HH = H1 @ H2
dist3 = get_mask(imgs[2], HH, h_max, w_max, channels)
alpha2 = dist12 / (dist12 + dist3)
dist2 = alpha2[..., np.newaxis] * dst1 + (1 - alpha2[..., np.newaxis]) * warp(
    imgs[2], HH, h_max, w_max, channels
)

output = dist2
np.clip(output, 0, 255, out=output)
output = output.astype(np.uint8)
cv2.imwrite("output_alpha.png", output)

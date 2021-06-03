import numpy as np
import cv2
import cv2.ximgproc as xip

d = 12
sigmaSpace = 15
sigmaColor = 20


def computeLocalBinaryPattern(Img, windowSize=7):
    R = windowSize // 2
    census = np.zeros_like(Img, dtype=np.uint64)
    Img = cv2.copyMakeBorder(
        Img, top=R, left=R, right=R, bottom=R, borderType=cv2.BORDER_CONSTANT, value=0
    )
    h, w, _ = Img.shape
    center = Img[R : h - R, R : w - R, :]
    offsets = [
        (r, c) for r in range(windowSize) for c in range(windowSize) if not r == R == c
    ]
    for r, c in offsets:
        census |= Img[r : r + h - R * 2, c : c + w - R * 2, :] >= center
        census <<= 1
    return census


def popcount64(mat):
    mat[...] = (mat & 0x5555555555555555) + ((mat & 0xAAAAAAAAAAAAAAAA) >> 1)
    mat[...] = (mat & 0x3333333333333333) + ((mat & 0xCCCCCCCCCCCCCCCC) >> 2)
    mat[...] = (mat & 0x0F0F0F0F0F0F0F0F) + ((mat & 0xF0F0F0F0F0F0F0F0) >> 4)
    mat[...] = (mat & 0x00FF00FF00FF00FF) + ((mat & 0xFF00FF00FF00FF00) >> 8)
    mat[...] = (mat & 0x0000FFFF0000FFFF) + ((mat & 0xFFFF0000FFFF0000) >> 16)
    mat[...] = (mat & 0x00000000FFFFFFFF) + (mat >> 32)
    return mat.astype(np.uint8)


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    binL = computeLocalBinaryPattern(Il)
    binR = computeLocalBinaryPattern(Ir)
    costL = np.zeros((max_disp + 1, h, w, ch), dtype=np.uint8)
    costR = np.zeros((max_disp + 1, h, w, ch), dtype=np.uint8)
    for disp in range(0, max_disp + 1):
        cost = popcount64(binL[:, disp:, :] ^ binR[:, : w - disp, :])
        # costL
        costL[disp, :, disp:, :] = cost
        costL[disp, :, :disp, :] = cost[:, np.newaxis, 0, :]
        # costR
        costR[disp, :, : w - disp, :] = cost
        costR[disp, :, w - disp :, :] = cost[:, np.newaxis, -1, :]

        # >>> Cost Aggregation
        # TODO: Refine the cost according to nearby costs
        # [Tips] Joint bilateral filter (for the cost of each disparty)
        xip.jointBilateralFilter(
            Il,
            costL[disp, ...],
            dst=costL[disp, ...],
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace,
        )
        xip.jointBilateralFilter(
            Ir,
            costR[disp, ...],
            dst=costR[disp, ...],
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace,
        )

    # hamming distance is at most 8 for each pixel per channel
    # When 8*ch < 256, i.e. ch < 32, summing cost over channels won't overflow
    costL = costL.sum(axis=-1)
    costR = costR.sum(axis=-1)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    dispL = np.argmin(costL, axis=0)
    dispR = np.argmin(costR, axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    newX = x - dispL
    valid = (dispL == dispR[y, newX]) & (newX >= 0)
    invalid = ~valid

    fL = dispL.copy()
    fR = dispL.copy()
    # TODO Pad maximum for holes in the boundary

    idx = np.where(valid, np.arange(0, w), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    fL[invalid] = fL[np.nonzero(invalid)[0], idx[invalid]]

    idx2 = np.where(valid, np.arange(0, w), 0)
    np.maximum.accumulate(idx2[::-1], axis=1, out=idx2[::-1])
    fR[invalid] = fR[np.nonzero(invalid)[0], idx2[invalid]]

    filled = np.minimum(fL, fR).astype(np.uint8)
    xip.weightedMedianFilter(Il, filled, dst=labels, r=5)
    return labels

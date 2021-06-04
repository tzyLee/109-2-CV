import numpy as np
import cv2
import cv2.ximgproc as xip

d = 12
sigmaSpace = 15
sigmaColor = 20
lambdaCost = 400
penalty1 = 0.01
penalty2Init = 2


def computeLocalBinaryPattern(Img, windowSize=5):
    R = windowSize // 2
    census = np.zeros_like(Img, dtype=np.uint64)
    Img = cv2.copyMakeBorder(
        Img, top=R, left=R, right=R, bottom=R, borderType=cv2.BORDER_CONSTANT, value=0
    )
    h, w, _ = Img.shape
    center = Img[R : h - R, R : w - R, :]
    for r in range(windowSize):
        for c in range(windowSize):
            if r == c == R:
                continue
            census <<= 1
            census |= Img[r : r + h - R * 2, c : c + w - R * 2, :] >= center
    return census


def popcount64(mat):
    mat[...] = (mat & 0x5555555555555555) + ((mat & 0xAAAAAAAAAAAAAAAA) >> 1)
    mat[...] = (mat & 0x3333333333333333) + ((mat & 0xCCCCCCCCCCCCCCCC) >> 2)
    mat[...] = (mat & 0x0F0F0F0F0F0F0F0F) + ((mat & 0xF0F0F0F0F0F0F0F0) >> 4)
    mat[...] = (mat & 0x00FF00FF00FF00FF) + ((mat & 0xFF00FF00FF00FF00) >> 8)
    mat[...] = (mat & 0x0000FFFF0000FFFF) + ((mat & 0xFFFF0000FFFF0000) >> 16)
    mat[...] = (mat & 0x00000000FFFFFFFF) + (mat >> 32)
    return mat


def costAggregate4(Img, cost):
    nDisp, h, w = cost.shape

    # Perform 4-direction DP-based cost aggregation
    # Left to center (West)
    aggCostW = np.zeros_like(cost, dtype=np.float32)
    minCostW = np.full((nDisp + 2, h), np.inf, dtype=np.float32)

    minCostW[1:-1, :] = cost[:, :, 0]
    aggCostW[:, :, 0] = cost[:, :, 0]
    minAggCostW = minCostW.min(axis=0)
    for i in range(1, w):
        minCostW[1:-1, :] = (
            cost[:, :, i]
            + np.minimum(
                np.minimum(minCostW[:-2, :], minCostW[2:, :]) + penalty1,
                np.minimum(
                    minCostW[1:-1, :],
                    minAggCostW
                    + penalty2Init
                    / (np.abs(Img[:, i, :] - Img[:, i - 1, :]).sum(axis=-1) + 1e-6),
                ),
            )
            - minAggCostW
        )
        aggCostW[:, :, i] = minCostW[1:-1, :]
        minCostW.min(axis=0, out=minAggCostW)

    # Right to center (East)
    aggCostE = np.zeros_like(cost, dtype=np.float32)
    minCostE = np.full((nDisp + 2, h), np.inf, dtype=np.float32)

    minCostE[1:-1, :] = cost[:, :, -1]
    aggCostE[:, :, -1] = cost[:, :, -1]
    minAggCostE = minCostE.min(axis=0)
    for i in reversed(range(w - 1)):
        minCostE[1:-1, :] = (
            cost[:, :, i]
            + np.minimum(
                np.minimum(minCostE[:-2, :], minCostE[2:, :]) + penalty1,
                np.minimum(
                    minCostE[1:-1, :],
                    minAggCostE
                    + penalty2Init
                    / (np.abs(Img[:, i, :] - Img[:, i + 1, :]).sum(axis=-1) + 1e-16),
                ),
            )
            - minAggCostE
        )
        aggCostE[:, :, i] = minCostE[1:-1, :]
        minCostE.min(axis=0, out=minAggCostE)

    # Top to center (North)
    aggCostN = np.zeros_like(cost, dtype=np.float32)
    minCostN = np.full((nDisp + 2, w), np.inf, dtype=np.float32)

    minCostN[1:-1, :] = cost[:, 0, :]
    aggCostN[:, 0, :] = cost[:, 0, :]
    minAggCostN = minCostN.min(axis=0)
    for i in range(1, h):
        minCostN[1:-1, :] = (
            cost[:, i, :]
            + np.minimum(
                np.minimum(minCostN[:-2, :], minCostN[2:, :]) + penalty1,
                np.minimum(
                    minCostN[1:-1, :],
                    minAggCostN
                    + penalty2Init
                    / (np.abs(Img[i, :, :] - Img[i - 1, :, :]).sum(axis=-1) + 1e-16),
                ),
            )
            - minAggCostN
        )
        aggCostN[:, i, :] = minCostN[1:-1, :]
        minCostN.min(axis=0, out=minAggCostN)

    # Bottom to center (South)
    aggCostS = np.zeros_like(cost, dtype=np.float32)
    minCostS = np.full((nDisp + 2, w), np.inf, dtype=np.float32)

    minCostS[1:-1, :] = cost[:, -1, :]
    aggCostS[:, -1, :] = cost[:, -1, :]
    minAggCostS = minCostS.min(axis=0)
    for i in reversed(range(h - 1)):
        minCostS[1:-1, :] = (
            cost[:, i, :]
            + np.minimum(
                np.minimum(minCostS[:-2, :], minCostS[2:, :]) + penalty1,
                np.minimum(
                    minCostS[1:-1, :],
                    minAggCostS
                    + penalty2Init
                    / (np.abs(Img[i, :, :] - Img[i + 1, :, :]).sum(axis=-1) + 1e-16),
                ),
            )
            - minAggCostS
        )
        aggCostS[:, i, :] = minCostS[1:-1, :]
        minCostS.min(axis=0, out=minAggCostS)

    return aggCostW + aggCostE + aggCostN + aggCostS


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    Ilf = Il.astype(np.float32)
    Irf = Ir.astype(np.float32)

    # >>> Cost Computation
    # Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    binL = computeLocalBinaryPattern(Il)
    binR = computeLocalBinaryPattern(Ir)
    costL = np.zeros((max_disp + 1, h, w), dtype=np.float32)
    costR = np.zeros((max_disp + 1, h, w), dtype=np.float32)
    for disp in range(0, max_disp + 1):
        # hamming distance is at most 8 for each pixel per channel
        # When 8*ch < 256, i.e. ch < 32, summing cost over channels won't overflow
        cost = popcount64(binL[:, disp:, :] ^ binR[:, : w - disp, :]).astype(np.float32)
        cost = 1 - np.exp(-cost / lambdaCost)
        cost = cost.sum(axis=-1)

        # > costL
        costL[disp, :, disp:] = cost
        costL[disp, :, :disp] = cost[:, np.newaxis, 0]

        # > costR
        costR[disp, :, : w - disp] = cost
        costR[disp, :, w - disp :] = cost[:, np.newaxis, -1]

    # >>> Cost Aggregation
    # Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    aggCostL = costAggregate4(Ilf, costL)
    aggCostR = costAggregate4(Irf, costR)

    for disp in range(0, max_disp + 1):
        xip.jointBilateralFilter(
            Ilf,
            aggCostL[disp, ...],
            dst=aggCostL[disp, ...],
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace,
        )
        xip.jointBilateralFilter(
            Irf,
            aggCostR[disp, ...],
            dst=aggCostR[disp, ...],
            d=d,
            sigmaColor=sigmaColor,
            sigmaSpace=sigmaSpace,
        )

    # >>> Disparity Optimization
    # Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    dispL = np.argmin(aggCostL, axis=0)
    dispR = np.argmin(aggCostR, axis=0)

    # >>> Disparity Refinement
    # Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    newX = x - dispL
    valid = (dispL == dispR[y, newX]) & (newX >= 0)

    # Pad maximum for holes in the boundary
    dispL = np.pad(dispL, pad_width=1, constant_values=max_disp)
    valid = np.pad(valid, pad_width=1, constant_values=True)
    invalid = ~valid

    fL = dispL.copy()
    fR = dispL.copy()
    # Fill holes from left to right
    idx = np.where(valid, np.arange(0, w + 2), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    fL[invalid] = fL[np.nonzero(invalid)[0], idx[invalid]]

    # Fill holes from right to left
    idx2 = np.where(valid[:, ::-1], np.arange(0, w + 2), 0)
    np.maximum.accumulate(idx2, axis=1, out=idx2)
    idx2 = w + 1 - idx2[:, ::-1]
    fR[invalid] = fR[np.nonzero(invalid)[0], idx2[invalid]]

    filled = np.minimum(fL, fR)[1:-1, 1:-1].astype(np.uint8)

    # Weighted median filtering
    smoothIl = cv2.bilateralFilter(
        Il, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
    )
    xip.weightedMedianFilter(smoothIl, filled, dst=labels, r=11)
    return labels

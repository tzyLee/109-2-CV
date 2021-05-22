import numpy as np
from scipy.interpolate import LinearNDInterpolator


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = np.zeros((9,))

    if v.shape[0] is not N:
        print("u and v should have the same size")
        return None
    if N < 4:
        print("At least 4 points should be given")

    UMean = np.mean(u, axis=0)
    UMaxStd = np.std(u, axis=0).max() + 1e-14
    u = u - UMean
    u /= UMaxStd
    S1 = np.array(
        [
            [1 / UMaxStd, 0, -UMean[0] / UMaxStd],
            [0, 1 / UMaxStd, -UMean[1] / UMaxStd],
            [0, 0, 1],
        ]
    )

    VMean = np.mean(v, axis=0)
    VMaxStd = np.std(v, axis=0).max() + 1e-14
    v = v - VMean
    v /= VMaxStd
    S2 = np.array(
        [
            [1 / VMaxStd, 0, -VMean[0] / VMaxStd],
            [0, 1 / VMaxStd, -VMean[1] / VMaxStd],
            [0, 0, 1],
        ]
    )

    # 1.forming A
    A = np.zeros((2 * N, 9))
    A[::2, :2] = u
    A[::2, 2] = 1
    A[::2, 6:8] = -v[:, 0, np.newaxis] * u
    A[::2, 8] = -v[:, 0]

    A[1::2, 3:5] = u
    A[1::2, 5] = 1
    A[1::2, 6:8] = -v[:, 1, np.newaxis] * u
    A[1::2, 8] = -v[:, 1]

    # 2.solve H with A
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape((3, 3))

    H = np.linalg.inv(S2) @ H @ S1
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction="b"):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # 1.meshgrid the (y, x) coordinate pairs
    coords = np.dstack(
        np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    ).reshape((-1, 2))

    N = coords.shape[0]
    # 2.reshape the destination pixels as N x 3 homogeneous coordinate
    u = np.hstack((coords, np.ones((N, 1), dtype=coords.dtype)))
    if direction == "b":
        H_inv_T = np.transpose(H_inv)
        # 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = u @ H_inv_T
        v[:, :2] /= v[:, 2, np.newaxis]
        v = v[:, :2]

        v0 = np.floor(v).astype(np.int32)
        v1 = v0 + 1
        # 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        xy_min = np.array([0, 0])
        xy_max = np.array([w_src, h_src])
        mask = (xy_min <= v).all(axis=1) & (v < xy_max).all(axis=1)
        # 5.sample the source image with the masked and reshaped transformed coordinates
        u = u[mask, :2]
        v0 = v0[mask, :]
        v1 = v1[mask, :]
        v = v[mask, :]
        x = v[:, 0]
        y = v[:, 1]
        x0 = v0[:, 0]
        y0 = v0[:, 1]
        x1 = v1[:, 0]
        y1 = v1[:, 1]
        np.clip(x0, 0, w_src - 1, out=x0)
        np.clip(y0, 0, h_src - 1, out=y0)
        np.clip(x1, 0, w_src - 1, out=x1)
        np.clip(y1, 0, h_src - 1, out=y1)

        dx0 = x - x0
        dy0 = y - y0
        dx1 = x1 - x
        dy1 = y1 - y

        wbr = dx0 * dy0
        wtr = dx0 * dy1
        wtl = dx1 * dy1
        wbl = dx1 * dy0

        interpolated = (
            wtl[..., None] * src[y0, x0, :]
            + wbl[..., None] * src[y1, x0, :]
            + wtr[..., None] * src[y0, x1, :]
            + wbr[..., None] * src[y1, x1, :]
        )
        np.clip(interpolated, 0, 255, out=interpolated)
        # 6. assign to destination image with proper masking
        dst[u[:, 1], u[:, 0], :] = interpolated.astype(dst.dtype)
    elif direction == "f":
        H_T = np.transpose(H)
        # 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        v = u @ H_T
        v[:, :2] /= v[:, 2, np.newaxis]
        v = v[:, :2]
        np.around(v, out=v)
        # 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        xy_min = np.array([0, 0])
        xy_max = np.array([w_dst, h_dst])
        mask = (xy_min <= v).all(axis=1) & (v < xy_max).all(axis=1)
        # 5.filter the valid coordinates using previous obtained mask
        u = u[mask, :2]
        v = v[mask, :].astype(np.int32)
        # 6. assign to destination image using advanced array indicing
        dst[v[:, 1], v[:, 0], :] = src[u[:, 1], u[:, 0], :]

    return dst

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
        np.around(v, out=v)
        # 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        xy_min = np.array([0, 0])
        xy_max = np.array([w_src, h_src])
        mask = (xy_min <= v).all(axis=1) & (v < xy_max).all(axis=1)
        # 5.sample the source image with the masked and reshaped transformed coordinates
        u = u[mask, :2]
        v = v[mask, :].astype(np.int32)
        # 6. assign to destination image with proper masking
        dst[u[:, 1], u[:, 0], :] = src[v[:, 1], v[:, 0], :]
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

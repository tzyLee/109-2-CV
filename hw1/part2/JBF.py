import numpy as np
import cv2
from cv2 import absdiff


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = ws = 6 * sigma_s + 1
        self.pad_w = r = 3 * sigma_s

        # Pre-compute range kernel value (Gaussian(0^2, sigma_r) ~ Gaussian(255^2, sigma_r))
        cache = np.arange(256) / 255
        cache **= 2
        cache /= -2 * sigma_r * sigma_r
        np.exp(cache, out=cache)
        self.cache = cache

        # Pre-compute spatial kernel
        ind = (np.arange(ws) - r) ** 2
        kernel_s = -(ind + ind[:, np.newaxis]) / (2 * sigma_s * sigma_s)
        np.exp(kernel_s, out=kernel_s)
        self.kernel_s = kernel_s

        self.offsets = [divmod(i, ws) for i in range(ws * ws)]

    def joint_bilateral_filter(self, img, guidance):
        r = self.pad_w
        ws = self.wndw_size
        kernel_s = self.kernel_s
        cache = self.cache
        offsets = self.offsets

        # Image padding
        BORDER_TYPE = cv2.BORDER_REFLECT
        I = cv2.copyMakeBorder(img, r, r, r, r, BORDER_TYPE)
        G = cv2.copyMakeBorder(guidance, r, r, r, r, BORDER_TYPE)
        h, w, ch = img.shape

        output = np.zeros_like(img, dtype=np.float64)
        den = np.zeros((h, w))

        # Main body
        if I.ndim == 3 and G.ndim == 2:
            for y, x in offsets:
                # (h, w)
                kernel = (
                    cache[absdiff(G[y : y + h, x : x + w], guidance)] * kernel_s[y, x]
                )
                output += kernel[..., None] * I[y : y + h, x : x + w, :]
                den += kernel
            output /= den[..., None]
        elif I.ndim == 3 and G.ndim == 3:
            for y, x in offsets:
                # (h, w)
                kernel = (
                    cache[absdiff(G[y : y + h, x : x + w, :], guidance)].prod(axis=-1)
                    * kernel_s[y, x]
                )
                output += kernel[..., None] * I[y : y + h, x : x + w, :]
                den += kernel
            output /= den[..., None]
        np.clip(output, 0, 255, out=output)
        return output.astype(np.uint8)

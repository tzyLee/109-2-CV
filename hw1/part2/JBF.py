import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = ws = 6 * sigma_s + 1
        self.pad_w = r = 3 * sigma_s

        # Pre-compute range kernel value (Gaussian(-255^2, sigma_r) ~ Gaussian(255^2, sigma_r))
        cache = (np.arange(255 * 2 + 1) - 255) / 255
        cache **= 2
        cache /= -2 * sigma_r * sigma_r
        np.exp(cache, out=cache)
        self.cache = cache

        # Pre-compute spatial kernel
        ind = (np.arange(ws) - r) ** 2
        kernel_s = -(ind + ind[:, np.newaxis]) / (2 * sigma_s * sigma_s)
        np.exp(kernel_s, out=kernel_s)
        self.kernel_s = kernel_s[..., np.newaxis, np.newaxis]

    def joint_bilateral_filter(self, img, guidance):
        # Add padding
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        )
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        )

        # Force the shape to be (H, W, ch)
        if padded_guidance.ndim == 2:
            padded_guidance = padded_guidance[..., np.newaxis]

        ws = self.wndw_size
        r = self.pad_w
        # Extract the view of shape (ws, ws, H, W, ch)
        h, w, ch = padded_img.shape
        img_blocks = np.lib.stride_tricks.as_strided(
            padded_img,
            shape=(ws, ws, *img.shape),
            strides=(w * ch, ch, w * ch, ch, 1),
            writeable=False,
        )

        ch = padded_guidance.shape[2]
        guidance_blocks = np.lib.stride_tricks.as_strided(
            padded_guidance,
            shape=(ws, ws, *img.shape[:2], ch),
            strides=(w * ch, ch, w * ch, ch, 1),
            writeable=False,
        )

        # range part, multiply over all channels
        # shape = (ws, ws, H, W)
        diff = guidance_blocks.astype(np.int16)
        diff -= guidance_blocks[r, r, ...]
        kernel = (self.cache[diff + 255].prod(axis=-1)) * self.kernel_s

        # shape = (H, W, ch)
        output = (kernel[..., np.newaxis] * img_blocks).sum(axis=(0, 1)) / (
            kernel.sum(axis=(0, 1))[..., np.newaxis]
        )

        np.clip(output, 0, 255, out=output)
        return output.astype(np.uint8)

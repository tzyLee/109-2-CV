import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(
        description="main function of joint bilateral filter"
    )
    parser.add_argument(
        "--image_path", default="./testdata/1.png", help="path to input image"
    )
    parser.add_argument(
        "--setting_path",
        default="./testdata/1_setting.txt",
        help="path to setting file",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(args.setting_path, "r") as f:
        points = []
        sigma_r = sigma_s = 0

        lines = iter(f)
        next(lines)
        for line in f:
            vals = line.strip().split(",")
            if vals[0].startswith("sigma"):
                sigma_s = int(vals[1])
                sigma_r = float(vals[3])
            else:
                points.append([float(num) for num in vals])

    jbf = Joint_bilateral_filter(sigma_s, sigma_r)

    names = ["cv2_gray"]
    names.extend(f"{r}_{g}_{b}_gray" for r, g, b in points)
    # (N, 1, 1, 3), the last axis is reversed (r, g, b) -> (b, g, r)
    weights = np.array(points)[:, np.newaxis, np.newaxis, ::-1]

    # (N+1, h, w)
    gray = np.concatenate(
        (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[np.newaxis, ...],
            # (N, h, w, 3) -> (N, h, w)
            (img * weights).sum(axis=-1).astype(np.uint8),
        )
    )

    bf = jbf.joint_bilateral_filter(img, img)
    # (N+1, h, w, 3)
    jbf_out = np.zeros((len(names), *img.shape), dtype=np.uint8)
    for i in range(len(names)):
        jbf_out[i, ...] = jbf.joint_bilateral_filter(img, gray[i, ...])
    l1_norm = jbf_out.astype(np.int32) - bf
    np.abs(l1_norm, out=l1_norm)
    diff = l1_norm.sum(axis=(1, 2, 3))

    # Show and save output
    cv2.imshow("bf", bf)
    cv2.imwrite(f"{args.image_path}_bf.png", bf)

    max_ind = np.argmax(diff)
    min_ind = np.argmin(diff)
    for i, name in enumerate(names):
        cv2.imshow(name, gray[i, ...])
        cv2.imwrite(f"{args.image_path}_{name}.png", gray[i, ...])

        cv2.imshow(f"{name}_jbf", jbf_out[i, ...])
        cv2.imwrite(f"{args.image_path}_{name}_jbf.png", jbf_out[i, ...])

        print(f"{name}: {diff[i]}", end="")
        if i == max_ind:
            print(" (max)")
        elif i == min_ind:
            print(" (min)")
        else:
            print()

    key = cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
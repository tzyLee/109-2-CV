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
    bf = jbf.joint_bilateral_filter(img, img)
    cv2.imshow("bf", bf)

    grays = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grays.append(gray)
    cv2.imshow("cv2 gray", gray)
    cv2.imwrite(f"{args.image_path}_cv2_gray.png", gray)

    for r, g, b in points:
        weights = np.array([b, g, r])
        gray = (img * weights).sum(axis=-1).astype(np.uint8)
        grays.append(gray)
        cv2.imshow(f"r={r} g={g} b={b}", gray)
        cv2.imwrite(f"{args.image_path}_{r}_{g}_{b}.png", gray)

    minInd = 0
    minDiff = float("inf")
    for ind, gray in enumerate(grays):
        out = jbf.joint_bilateral_filter(img, gray)
        l1_norm = out.astype(np.int32) - bf
        np.abs(l1_norm, out=l1_norm)

        diff = l1_norm.sum()
        if diff < minDiff:
            minDiff = diff
            minInd = ind

    if minInd == 0:
        print("best weights are default weights")
    else:
        print("best weights are", points[minInd - 1])
    key = cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
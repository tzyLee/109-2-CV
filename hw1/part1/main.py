import numpy as np
import cv2
import argparse
from HCD import Harris_corner_detector


def main():
    parser = argparse.ArgumentParser(
        description="main function of Harris corner detector"
    )
    parser.add_argument(
        "--threshold",
        default=100.0,
        type=float,
        help="threshold value to determine corner",
    )
    parser.add_argument(
        "--image_path", default="./testdata/1.png", help="path to input image"
    )
    args = parser.parse_args()

    print("Processing %s ..." % args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    hcd = Harris_corner_detector(args.threshold)
    response = hcd.detect_harris_corners(img_gray)
    result = hcd.post_processing(response)

    for y, x in result:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("result", img)

    key = cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(f"{args.image_path}_{args.threshold}.png", img)


if __name__ == "__main__":
    main()
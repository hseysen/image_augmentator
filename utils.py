import numpy as np
import cv2
from helpers import *


# TODO: Add more augmentation methods


def augmentate_rotation(image, annotations, angle=45):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_angle = angle * np.pi / 180

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_img = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    new_height, new_width = rotated_img.shape[:2]

    rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                           [np.sin(rotation_angle), np.cos(rotation_angle)]])

    new_bbox = []
    for bbox in annotations:
        if len(bbox) > 1:
            (center_x, center_y, bbox_width, bbox_height) = yolotocv(bbox[1], bbox[2], bbox[3], bbox[4], height, width)

            upper_left_corner_shift = (center_x - width / 2, -height / 2 + center_y)
            upper_right_corner_shift = (bbox_width - width / 2, -height / 2 + center_y)
            lower_left_corner_shift = (center_x - width / 2, -height / 2 + bbox_height)
            lower_right_corner_shift = (bbox_width - width / 2, -height / 2 + bbox_height)

            new_lower_right_corner = [-1, -1]
            new_upper_left_corner = []

            for i in (upper_left_corner_shift, upper_right_corner_shift,
                      lower_left_corner_shift, lower_right_corner_shift):
                new_coords = np.matmul(rot_matrix, np.array((i[0], -i[1])))
                x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                if new_lower_right_corner[0] < x_prime:
                    new_lower_right_corner[0] = x_prime
                if new_lower_right_corner[1] < y_prime:
                    new_lower_right_corner[1] = y_prime

                if len(new_upper_left_corner) > 0:
                    if new_upper_left_corner[0] > x_prime:
                        new_upper_left_corner[0] = x_prime
                    if new_upper_left_corner[1] > y_prime:
                        new_upper_left_corner[1] = y_prime
                else:
                    new_upper_left_corner.append(x_prime)
                    new_upper_left_corner.append(y_prime)

            new_bbox.append([int(bbox[0]), *cvtoyolo(new_upper_left_corner[0], new_upper_left_corner[1],
                             new_lower_right_corner[0], new_lower_right_corner[1], new_height, new_width)])

    return rotated_img, new_bbox, new_height, new_width


def augmentate_flip(image, annotations, flipdir="h"):
    assert flipdir in ["h", "v"]
    if flipdir == "h":
        flipped_img = cv2.flip(image, 1)
    else:
        flipped_img = cv2.flip(image, 0)

    new_bbox = []
    for bbox in annotations:
        c, x, y, w, h = bbox
        if len(bbox) > 1:
            if flipdir == "h":
                x = round(1 - x, 6)
            if flipdir == "v":
                y = round(1 - y, 6)
            new_bbox.append([int(c), x, y, w, h])

    return flipped_img, new_bbox


def augmentate_saltnpeppernoise(image, annotations, noise_intensity=0.04):
    black = np.array([0, 0, 0], dtype="uint8")
    white = np.array([255, 255, 255], dtype="uint8")
    probs = np.random.random(image.shape[:2])
    image[probs < (noise_intensity / 2)] = black
    image[probs > 1 - (noise_intensity / 2)] = white
    return image, annotations


def augmentate_bilateral(image, annotations, dist=23, scolor=98, sspace=88):
    return cv2.bilateralFilter(image, dist, scolor, sspace), annotations


def augmentate_gaussianblur(image, annotations, kw=7, kh=7, sigma=25):
    return cv2.GaussianBlur(image, (kw, kh), sigma), annotations


def draw_annotations(starting_img, annotations_to_draw, col, thk):
    nh, nw = starting_img.shape[:2]
    drawn_img = starting_img
    for a in annotations_to_draw:
        obj, x, y, w, h = map(float, a)
        x1, y1, x2, y2 = yolotocv(x, y, w, h, nh, nw)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        drawn_img = cv2.rectangle(drawn_img, p1, p2, col, thk)
    return drawn_img


def main():
    # Configuration
    img_dir = "./images/input_image.jpg"
    ann_dir = "./annotations/input_image.txt"
    color = (0, 122, 255)
    thickness = 4

    # Test certain augmentations
    test_rotation = False
    test_flip = False
    test_saltnpepper = False
    test_bilateral = False
    test_gaussianblur = False

    # Load data
    img = cv2.imread(img_dir)
    with open(ann_dir, "r") as f:
        original_anns = [list(map(float, a.strip('\n').split(' '))) for a in f.readlines()]

    # Perform augmentations for testing
    if test_rotation:
        for rot_angle in np.linspace(0, 360, 15):
            rotated, ann, _, _ = augmentate_rotation(img, original_anns, rot_angle)
            rotated = draw_annotations(rotated, ann, color, thickness)
            cv2.imshow(f"Image Rotate - {rot_angle}", rotated)
            cv2.waitKey(0)

    if test_flip:
        for flip_direction in ["h", "v"]:
            flipped, ann = augmentate_flip(img, original_anns, flip_direction)
            flipped = draw_annotations(flipped, ann, color, thickness)
            cv2.imshow(f"Image Flip - {flip_direction}", flipped)
            cv2.waitKey(0)

    if test_saltnpepper:                        # REMINDER: Noise intensity should be generally low
        for noise_int in np.linspace(0, 0.003, 25):
            noisy, ann = augmentate_saltnpeppernoise(img, original_anns, noise_int)
            noisy = draw_annotations(noisy, ann, color, thickness)
            cv2.imshow(f"Image Salt and Pepper - {noise_int}", noisy)
            cv2.waitKey(0)

    if test_bilateral:
        for d in range(10, 20):
            for c in range(0, 100, 25):
                for s in range(0, 100, 25):
                    bil, ann = augmentate_bilateral(img, original_anns, d, c, s)
                    bil = draw_annotations(bil, ann, color, thickness)
                    cv2.imshow(f"Image Bilateral - {d} {c} {s}", bil)
                    cv2.waitKey(0)

    if test_gaussianblur:                       # REMINDER: Kernel dimensions should be odd numbers
        for w in range(1, 17, 4):
            for h in range(1, 17, 4):
                for s in range(0, 100, 50):
                    gaus, ann = augmentate_gaussianblur(img, original_anns, w, h, s)
                    gaus = draw_annotations(gaus, ann, color, thickness)
                    cv2.imshow(f"Image Gaussian - {w} {h} {s}", gaus)
                    cv2.waitKey(0)


if __name__ == "__main__":
    main()

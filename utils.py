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


def draw_annotations(starting_img, annotations_to_draw, c, t):
    nh, nw = starting_img.shape[:2]
    drawn_img = starting_img
    for a in annotations_to_draw:
        obj, x, y, w, h = map(float, a)
        x1, y1, x2, y2 = yolotocv(x, y, w, h, nh, nw)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        drawn_img = cv2.rectangle(drawn_img, p1, p2, c, t)
    return drawn_img


if __name__ == "__main__":
    img_dir = "./images/input_image.jpg"
    ann_dir = "./annotations/input_image.txt"
    color = (0, 122, 255)
    thickness = 4
    TEST_ROTATION = True
    TEST_FLIP = False

    img = cv2.imread(img_dir)
    with open(ann_dir, "r") as f:
        original_anns = [list(map(float, a.strip('\n').split(' '))) for a in f.readlines()]

    if TEST_ROTATION:
        for rot_angle in np.linspace(0, 360, 15):
            rotated, ann, _, _ = augmentate_rotation(img, original_anns, rot_angle)
            rotated = draw_annotations(rotated, ann, color, thickness)
            cv2.imshow("Image", rotated)
            cv2.waitKey(0)

    if TEST_FLIP:
        for flip_direction in ["h", "v"]:
            flipped, ann = augmentate_flip(img, original_anns, flip_direction)
            flipped = draw_annotations(flipped, ann, color, thickness)
            cv2.imshow("Image", flipped)
            cv2.waitKey(0)

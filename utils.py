import numpy as np
import cv2
import random
from helpers import *


COLOR = (0, 122, 255)
THICKNESS = 4


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
            (x1, y1, x2, y2) = yolotocv(bbox[1], bbox[2], bbox[3], bbox[4], height, width)

            ul_corner_shift = (x1 - width / 2, y1 - height / 2)
            ur_corner_shift = (x2 - width / 2, y1 - height / 2)
            ll_corner_shift = (x1 - width / 2, y2 - height / 2)
            lr_corner_shift = (x2 - width / 2, y2 - height / 2)

            new_lr_corner = [-1, -1]
            new_ul_corner = []

            for i in (ul_corner_shift, ur_corner_shift,
                      ll_corner_shift, lr_corner_shift):
                new_coords = np.matmul(rot_matrix, np.array((i[0], -i[1])))
                x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                if new_lr_corner[0] < x_prime:
                    new_lr_corner[0] = x_prime
                if new_lr_corner[1] < y_prime:
                    new_lr_corner[1] = y_prime

                if len(new_ul_corner) > 0:
                    if new_ul_corner[0] > x_prime:
                        new_ul_corner[0] = x_prime
                    if new_ul_corner[1] > y_prime:
                        new_ul_corner[1] = y_prime
                else:
                    new_ul_corner.append(x_prime)
                    new_ul_corner.append(y_prime)

            new_bbox.append([int(bbox[0]), *cvtoyolo(new_ul_corner[0], new_ul_corner[1],
                                                     new_lr_corner[0], new_lr_corner[1], new_height,
                                                     new_width)])

    return rotated_img, new_bbox, new_height, new_width


def augmentate_perspective(image, annotations, dx1, dx2, dy1, dy2):
    height, width = image.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[dx1, dy1], [width-dx1, dy2], [dx2, height-dy1], [width-dx2, height-dy2]])
    m = cv2.getPerspectiveTransform(pts1, pts2)

    image = cv2.warpPerspective(image, m, (width, height))

    new_bbox = []
    for bbox in annotations:
        (x_a, y_a, x_b, y_b) = yolotocv(bbox[1], bbox[2], bbox[3], bbox[4], height, width)

        rect_pts = np.array([[[x_a, y_a]], [[x_b, y_a]], [[x_a, y_b]], [[x_b, y_b]]], dtype=np.float32)
        new_rect = cv2.perspectiveTransform(rect_pts, m)

        new_x_a = new_rect[0][0][0]
        new_x_b = new_rect[1][0][0]
        new_y_a = new_rect[0][0][1]
        new_y_b = new_rect[2][0][1]

        new_bbox.append([int(bbox[0]), *cvtoyolo(new_x_a, new_y_a, new_x_b, new_y_b, height, width)])

    return image, new_bbox


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


def augmentate_saltnpeppernoise(image, annotations, noise_intensity):
    black = np.array([0, 0, 0], dtype="uint8")
    white = np.array([255, 255, 255], dtype="uint8")
    probs = np.random.random(image.shape[:2])
    img = image.copy()                                          # just in case
    img[probs < (noise_intensity / 2)] = black
    img[probs > 1 - (noise_intensity / 2)] = white
    return img, annotations


def augmentate_bilateral(image, annotations, dist, scolor, sspace):
    return cv2.bilateralFilter(image, dist, scolor, sspace), annotations


def augmentate_gaussianblur(image, annotations, kw, kh, sigma):
    return cv2.GaussianBlur(image, (kw, kh), sigma), annotations


def augmentate_shift(image, annotations, tx, ty, minobjsize=0.001):
    height = image.shape[0]
    width = image.shape[1]
    mx = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    new_ann = []
    for obj, cx, cy, wo, ho in annotations:
        old_ann = yolotocv(cx, cy, wo, ho, height, width)
        old_ann[0] = clamp_value(old_ann[0] + tx, 0, width)
        old_ann[1] = clamp_value(old_ann[1] + ty, 0, height)
        old_ann[2] = clamp_value(old_ann[2] + tx, 0, width)
        old_ann[3] = clamp_value(old_ann[3] + ty, 0, height)
        new_dims = cvtoyolo(*old_ann, height, width)
        if new_dims[2] > minobjsize and new_dims[3] > minobjsize:
            new_ann.append([obj] + cvtoyolo(*old_ann, height, width))

    shifted = cv2.warpAffine(image, mx, (width, height))
    return shifted, new_ann


def augmentate_hsv(image, annotations, dh, ds):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    h = np.mod(h + dh, 180).astype(np.uint8)
    s = np.clip(s * ds, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return image, annotations


def augmentate_contrast(image, annotations, gamma):
    lut = []
    for i in range(256):
        lut.append((i / 255) ** gamma * 255)
    lut = np.uint8(lut)
    image = cv2.LUT(image, lut)
    return image, annotations


def augmentate_sharpness(image, annotations, size, sigma):
    image = cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (size, size), sigma), -0.5, 0, image)
    return image, annotations


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


def test_augmentation(image, annotations, augmentation, *args):
    result = list(augmentation(image, annotations, *args))
    augmented = draw_annotations(result[0], result[1], COLOR, THICKNESS)
    cv2.imshow(f"Image {augmentation.__name__[augmentation.__name__.find('_'):]} - {args}", augmented)
    print(result[1])
    cv2.waitKey(0)


def main():
    # Configuration
    img_dir = "images/image0.jpg"
    ann_dir = "annotations/image0.txt"

    # Test certain augmentations
    test_rotation = False
    test_perspective = False
    test_flip = False
    test_saltnpepper = False
    test_bilateral = False
    test_gaussianblur = False
    test_shift = False
    test_hsv = False
    test_contrast = False
    test_sharpness = False

    # Load data
    img = cv2.imread(img_dir)
    with open(ann_dir, "r") as f:
        original_anns = [list(map(float, a.strip('\n').split(' '))) for a in f.readlines()]

    # Perform augmentations for testing
    if test_rotation:
        for rot_angle in np.linspace(0, 360, 15):
            test_augmentation(img, original_anns, augmentate_rotation, rot_angle)

    if test_perspective:
        for i in range(5):
            d1 = random.randint(1, 90)
            d2 = random.randint(1, 90)
            d3 = random.randint(1, 90)
            d4 = random.randint(1, 90)
            test_augmentation(img, original_anns, augmentate_perspective, d1, d2, d3, d4)

    if test_flip:
        for flip_direction in ["h", "v"]:
            test_augmentation(img, original_anns, augmentate_flip, flip_direction)

    if test_saltnpepper:  # REMINDER: Noise intensity should be generally low
        for noise_int in np.linspace(0, 0.003, 25):
            test_augmentation(img, original_anns, augmentate_saltnpeppernoise, noise_int)

    if test_bilateral:
        for d in range(10, 20, 2):
            for c in range(0, 100, 25):
                for s in range(0, 100, 25):
                    test_augmentation(img, original_anns, augmentate_bilateral, d, c, s)

    if test_gaussianblur:  # REMINDER: Kernel dimensions should be odd numbers
        for w in range(1, 17, 4):
            for h in range(1, 17, 4):
                for s in range(0, 100, 50):
                    test_augmentation(img, original_anns, augmentate_gaussianblur, w, h, s)

    if test_shift:
        for dx in range(-150, 151, 50):
            for dy in range(-150, 151, 50):
                test_augmentation(img, original_anns, augmentate_shift, dx, dy)

    if test_hsv:
        for deltah in np.linspace(0, 15, 7):
            for deltas in np.linspace(0, 3, 7):
                test_augmentation(img, original_anns, augmentate_hsv, deltah, deltas)

    if test_contrast:
        for g in [0.25, 0.33, 0.5, 1, 2, 3, 4]:
            test_augmentation(img, original_anns, augmentate_contrast, g)

    if test_sharpness:
        for ksize in [5, 7, 13, 19]:
            for sharpness_strength in [5, 100, 500]:
                test_augmentation(img, original_anns, augmentate_sharpness, ksize, sharpness_strength)


if __name__ == "__main__":
    main()

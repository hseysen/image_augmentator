import os
from utils import *


DRAW_BBOXES = True
BBOX_COLOR = (0, 122, 255)
BBOX_THICK = 4


def main():
    # TODO: add argparse
    folder_images = "./images"
    folder_anns = "./annotations"

    imgs = []
    for i in os.scandir(folder_images):
        imgs.append(i)

    anns = []
    for a in os.scandir(folder_anns):
        anns.append(a)

    for i, a in zip(imgs, anns):
        # Extract data from files and set up augmentation parameters
        # TODO: hook this up to argparse and perform range for each augmentation rather than one value
        original_img = cv2.imread(i.path)
        with open(a.path, "r") as f:
            original_anns = [list(map(float, a.strip('\n').split(' '))) for a in f.readlines()]
        target_angle = 45
        target_flip = "v"
        target_noise = 0.005
        # TODO: name files automatically and more clearly

        # Rotation augmentation
        rotated_img, rotated_ann, _, _ = augmentate_rotation(original_img, original_anns, target_angle)
        if DRAW_BBOXES:
            rotated_img = draw_annotations(rotated_img, rotated_ann, BBOX_COLOR, BBOX_THICK)
        cv2.imwrite(os.path.join(folder_images, f"rotated_{target_angle}_{i.name}"), rotated_img)
        with open(os.path.join(folder_anns, f"rotated_{target_angle}_{a.name}"), "w") as f:
            for ann in rotated_ann:
                f.write(" ".join(map(str, ann)) + "\n")

        # Flip augmentation on the rotated image to test scalability
        flipped_img, flipped_ann = augmentate_flip(rotated_img, rotated_ann, target_flip)
        if DRAW_BBOXES:
            flipped_img = draw_annotations(flipped_img, flipped_ann, BBOX_COLOR, BBOX_THICK)
        cv2.imwrite(os.path.join(folder_images, f"flipped_{target_flip}_rotated_{target_angle}_{i.name}"), flipped_img)
        with open(os.path.join(folder_anns, f"flipped_{target_flip}_rotated_{target_angle}_{a.name}"), "w") as f:
            for ann in flipped_ann:
                f.write(" ".join(map(str, ann)) + "\n")

        # Salt and pepper noise augmentation on the flipped image to test scalability
        noisy_img, noisy_ann = augmentate_saltnpeppernoise(flipped_img, flipped_ann, target_noise)
        if DRAW_BBOXES:
            noisy_img = draw_annotations(noisy_img, noisy_ann, BBOX_COLOR, BBOX_THICK)
        cv2.imwrite(os.path.join(folder_images, f"noisy_{int(target_noise*1000)}_flipped_{target_flip}_rotated_{target_angle}_{i.name}"), noisy_img)
        with open(os.path.join(folder_anns, f"noisy_{int(target_noise*1000)}_flipped_{target_flip}_rotated_{target_angle}_{a.name}"), "w") as f:
            for ann in noisy_ann:
                f.write(" ".join(map(str, ann)) + "\n")

        # TODO: Make a function for saving images+annotations to disk


if __name__ == "__main__":
    # Running this file should generate 2 files in /images and /annotations folders
    main()

import argparse
import os
import random
from utils import *


BBOX_COLOR = (0, 122, 255)
BBOX_THICK = 4


def save_to_disk(image, annotation, name, i_folder, a_folder):
    cv2.imwrite(os.path.join(i_folder, name + ".jpg"), image)
    with open(os.path.join(a_folder, name + ".txt"), "w") as f:
        for ann in annotation:
            f.write(" ".join(map(str, ann)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-images", type=str, required=False, default="./images",
                        help="The folder where image data are located. Defaults to ./images")
    parser.add_argument("--folder-anns", type=str, required=False, default="./annotations",
                        help="The folder where image data are located. Defaults to ./annotations")
    parser.add_argument("--rotate-min", type=float, required=False, default=0,
                        help="Minimum rotation angles. Defaults to 0")
    parser.add_argument("--rotate-max", type=float, required=False, default=180,
                        help="Maximum rotation angles. Defaults to 180")
    parser.add_argument("--flip", type=bool, required=False, default=True,
                        help="Whether or not to flip the images. True by default")
    parser.add_argument("--noise", type=float, required=False, default=0.005,
                        help="Maximum Salt & Pepper noise intensity. Defaults to 0.005")
    parser.add_argument("--bilateral", type=bool, required=False, default=False,
                        help="Whether or not to apply bilateral blurring to the images. False by default")
    parser.add_argument("--gaussian", type=bool, required=False, default=False,
                        help="Whether or not to apply gaussian blurring to the images. False by default")
    parser.add_argument("--hsv", type=bool, required=False, default=True,
                        help="Whether or not to apply color manipulation to the images. True by default")
    parser.add_argument("--shift-min", type=float, required=False, default=-30,
                        help="Minimum shift in pixels. Defaults to -30")
    parser.add_argument("--shift-max", type=float, required=False, default=30,
                        help="Maximum shift in pixels. Defaults to 30")
    parser.add_argument("--augs", type=int, required=False, default=5,
                        help="Maximum augmentations for each image. Defaults to 5")
    parser.add_argument("--draw-bbox", type=bool, required=False, default=False,
                        help="Whether or not to draw bounding boxes on the images at the end. False by default")

    args = parser.parse_args()

    imgs = []
    for i in os.scandir(args.folder_images):
        imgs.append(i)

    anns = []
    for a in os.scandir(args.folder_anns):
        anns.append(a)

    for i, a in zip(imgs, anns):
        fname = i.name[:i.name.rfind(".")]

        # Extract data from files and set up augmentation parameters
        original_img = cv2.imread(i.path)
        with open(a.path, "r") as f:
            original_anns = [list(map(float, a.strip('\n').split(' '))) for a in f.readlines()]
        # TODO: name files automatically and more clearly

        augs_for_this = random.randint(1, args.augs)

        for aug_iter in range(augs_for_this):
            new_img = original_img.copy()
            new_ann = original_anns.copy()

            # Rotation augmentation
            target_angle = random.random() * (args.rotate_max - args.rotate_min) + args.rotate_min
            new_img, new_ann, _, _ = augmentate_rotation(new_img, new_ann, target_angle)

            # Flip augmentation
            if args.flip:
                target_flip = random.choice(["h", "v"])
                new_img, new_ann = augmentate_flip(new_img, new_ann, target_flip)

            # Bilateral Blur augmentation
            if args.bilateral:
                target_dist = int(random.random() * 23)
                target_scolor = int(random.random() * 98)
                target_sspace = int(random.random() * 88)
                new_img, new_ann = augmentate_bilateral(new_img, new_ann, target_dist, target_scolor, target_sspace)

            # Gaussian Blur augmentation
            if args.gaussian:
                target_kw = int(random.randrange(1, 9, 2))
                target_kh = int(random.randrange(1, 9, 2))
                target_sigma = random.random() * 25
                new_img, new_ann = augmentate_gaussianblur(new_img, new_ann, target_kw, target_kh, target_sigma)

            # HSV augmentation
            if args.hsv:
                target_dh = random.random() * 15 + 0.85
                target_ds = random.random() * 3 + 0.5
                new_img, new_ann = augmentate_hsv(new_img, new_ann, target_dh, target_ds)

            # Shift augmentation
            target_shift_x = random.random() * (args.shift_max - args.shift_min) + args.shift_min
            target_shift_y = random.random() * (args.shift_max - args.shift_min) + args.shift_min
            new_img, new_ann = augmentate_shift(new_img, new_ann, target_shift_x, target_shift_y)

            # Salt and pepper noise augmentation
            target_noise = random.random() * args.noise
            new_img, new_ann = augmentate_saltnpeppernoise(new_img, new_ann, target_noise)

            # Drawing bounding boxes
            if args.draw_bbox:
                new_img = draw_annotations(new_img, new_ann, BBOX_COLOR, BBOX_THICK)
            save_to_disk(new_img, new_ann, fname + str(aug_iter), args.folder_images, args.folder_anns)


if __name__ == "__main__":
    # Running this file should generate 2 files in /images and /annotations folders
    main()

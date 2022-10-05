import argparse
import os
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
    parser.add_argument("--rotate-min", type=float, required=False, default=-90,
                        help="Minimum rotation angles. Defaults to -90")
    parser.add_argument("--rotate-max", type=float, required=False, default=90,
                        help="Maximum rotation angles. Defaults to 90")
    parser.add_argument("--perspective", action="store_true", default=False,
                        help="Use this flag to apply perspective transform to the images.")
    parser.add_argument("--flip", action="store_true", default=False,
                        help="Use this flag to flip the images.")
    parser.add_argument("--noise", type=float, required=False, default=0.001,
                        help="Maximum Salt & Pepper noise intensity. Defaults to 0.001")
    parser.add_argument("--bilateral", action="store_true", default=False,
                        help="Use this flag to apply bilateral blurring to the images.")
    parser.add_argument("--gaussian", action="store_true", default=False,
                        help="Use this flag to apply gaussian blurring to the images.")
    parser.add_argument("--hsv", action="store_true", default=False,
                        help="Use this flag to apply color manipulation to the images.")
    parser.add_argument("--contrast", action="store_true", default=False,
                        help="Use this flag to apply contrast manipulation to the images.")
    parser.add_argument("--sharpness", action="store_true", default=False,
                        help="Use this flag to apply sharpness manipulation to the images.")
    parser.add_argument("--shift-min", type=float, required=False, default=-10,
                        help="Minimum shift in pixels. Defaults to -10")
    parser.add_argument("--shift-max", type=float, required=False, default=10,
                        help="Maximum shift in pixels. Defaults to 10")
    parser.add_argument("--augs", type=int, required=False, default=5,
                        help="Maximum augmentations for each image. Defaults to 5")
    parser.add_argument("--draw-bbox", action="store_true", default=False,
                        help="Use this flag to draw bounding boxes on the images at the end.")

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

        augs_for_this = random.randint(1, args.augs)

        for aug_iter in range(augs_for_this):
            new_img = original_img.copy()
            new_ann = original_anns.copy()

            # Rotation augmentation
            target_angle = random.random() * (args.rotate_max - args.rotate_min) + args.rotate_min
            new_img, new_ann, _, _ = augmentate_rotation(new_img, new_ann, target_angle)

            # Perspective augmentation
            if args.perspective:
                target_dx1 = int(random.random() * 130)
                target_dx2 = int(random.random() * 130)
                target_dy1 = int(random.random() * 130)
                target_dy2 = int(random.random() * 130)
                new_img, new_ann = augmentate_perspective(new_img, new_ann,
                                                          target_dx1, target_dx2, target_dy1, target_dy2)

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
                target_dh = random.random() * 52 + 34
                target_ds = random.random() * 2.5 + 1.2
                new_img, new_ann = augmentate_hsv(new_img, new_ann, target_dh, target_ds)

            # Contrast augmentation
            if args.contrast:
                target_gamma = 0.75
                new_img, new_ann = augmentate_contrast(new_img, new_ann, target_gamma)

            # Sharpness augmentation
            if args.sharpness:
                target_ksize = random.choice([3, 5, 7, 9, 11, 13, 17, 19, 23])
                target_sigma = random.randint(5, 100)
                new_img, new_ann = augmentate_sharpness(new_img, new_ann, target_ksize, target_sigma)

            # Shift augmentation
            target_shift_x = random.random() * (args.shift_max - args.shift_min) + args.shift_min
            target_shift_y = random.random() * (args.shift_max - args.shift_min) + args.shift_min
            new_img, new_ann = augmentate_shift(new_img, new_ann, target_shift_x, target_shift_y)

            # Salt and pepper noise augmentation
            target_noise = random.random() * args.noise
            new_img, new_ann = augmentate_saltnpeppernoise(new_img, new_ann, target_noise)

            # Drawing bounding boxes and saving
            if args.draw_bbox:
                new_img = draw_annotations(new_img, new_ann, BBOX_COLOR, BBOX_THICK)
            save_to_disk(new_img, new_ann, fname + f"_augmented_{aug_iter}", args.folder_images, args.folder_anns)


if __name__ == "__main__":
    # Running this file should generate 2 files in /images and /annotations folders
    main()

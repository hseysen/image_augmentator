# Image Augmentator
## About
Just a handy script I've written to help me augmentate image data for training of Machine Learning models. Currently, only supports YOLO annotations, but might work on adding other formats as well in the future.

Following image augmentation methods are made available through this project:
* Rotation - rotates the image in degrees
* Perspective - applies perspective transform to the image
* Flip - flips the image vertically and/or horizontally
* Salt & Pepper - adds noise to the image
* Bilateral Blur - applies bilateral blur onto the image
* Gaussian Blur - applies gaussian blur onto the image
* Shift - shifts the image vertically and/or horizontally by predetermined number of pixels
* HSV - performs color manipulations on the image
* Contrast - modifies contrast/brightness property of the image

The augmentator aims to preserve the object annotations. To test the annotations bounding boxes, you can use `draw_annotations` function defined in `utils.py`.


## Installation
Clone the repository and create a Python virtual environment. I've worked with 3.8, but should work with other versions. Install the requirements from `requirements.txt`
```bash
pip install -r requirements.txt
```
Make sure to place your image data to `/images`, and your annotations to `/annotations`. I've only tested with `.jpg` images, but feel free to test with other formats. Annotations should have the same file name (not extension) as the images. Please take a copy of your data, just in case.

## Usage
Run the following command to see the command line arguments:
```bash
python augment.py -h
```
Descriptions of the arguments and their defaults are explained there. Feel free to experiment with different values, but defaults usually work the best. Running the file should give you the augmented images and annotations in your folders.
```bash
python augment.py [arguments]
```

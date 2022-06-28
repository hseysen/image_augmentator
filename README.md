# Image Augmentator
## About
Just a handy script I've written to help me augmentate image data for training of Machine Learning models. Currently, only supports YOLO annotations, but might work on adding other formats as well in the future.

For now, the augmentator only performs rotation and flipping, but more features will be added later.

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

# Image Augmentator
Just a handy script I've written to help me augmentate image data for training of Machine Learning models. Currently, only supports YoloV5 annotations, but might work on adding other formats as well in the future.

For now, the augmentator only performs rotation and flipping, but more features will be added later.

## Usage

Clone the repository and create a Python virtual environment. I've worked with 3.8, but should work with other versions. Install the requirements from `requirements.txt`

```bash
pip install -r requirements.txt
```

Then, just run the `augment.py` file, it should perform a test augmentation for you. Check the folders `annotations/` and `images/` after you've run the file.


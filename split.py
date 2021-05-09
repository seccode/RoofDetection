import argparse
import glob
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

# Data folders
LABELED_DATA_FOLDER = "imgs/labeled"
TRAIN_FOLDER = "imgs/train"
TEST_FOLDER = "imgs/test"

"""
Function to split train and test sets
"""
def split():

    # Get all images
    all_images = glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.jpg")) + \
                 glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.png"))

    imgs = sorted(all_images)
    # Get all labels
    labels = sorted(glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.npy")))

    # Reset the train and test folders to have no data
    for folder in (TRAIN_FOLDER, TEST_FOLDER):
        if os.path.isdir(folder):
            shutil.rmtree(folder)

        # Create clean folder
        os.mkdir(folder)

    # Split all data into train and test sets
    x_train, x_test, y_train, y_test = \
        train_test_split(np.array(imgs).reshape(-1, 1),
                         np.array(labels).reshape(-1, 1))

    # Repopulate the train and test folders with new split data
    for imgs, labels, folder in zip([x_train.flatten(), x_test.flatten()],
                            [y_train.flatten(), y_test.flatten()],
                            [TRAIN_FOLDER, TEST_FOLDER]):

        # Copy each image to given folder
        for img in imgs:
            shutil.copyfile(img, os.path.join(folder,
                                              os.path.basename(img)))

        # Copy each label to given folder
        for label in labels:
            shutil.copyfile(label, os.path.join(folder,
                                                os.path.basename(label)))

    return

if __name__ == "__main__":
    split()

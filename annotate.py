import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# Data folders
UNLABELED_DATA_FOLDER = "imgs/unlabeled"
LABELED_DATA_FOLDER = "imgs/labeled"
TMP_DATA_FOLDER = "imgs/tmp"

class Annotator:
    def __init__(self, img_file):
        assert UNLABELED_DATA_FOLDER in img_file, \
            "Img not in unlabeled folder"

        existing_imgs_count = len(
            glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.jpg")) +
            glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.png"))
        )

        self.img = plt.imread(img_file)
        self.height, self.width, _ = self.img.shape
        self.bboxes = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        plt.imshow(self.img)
        plt.show()

        # Save image to new location and create VOC xml file
        if len(self.bboxes) > 0:

            new_img_file = \
                os.path.join(
                    LABELED_DATA_FOLDER,
                    str(existing_imgs_count + 1) + ".png"
                )

            os.rename(
                img_file,
                new_img_file
            )

            np.save(
                os.path.join(
                    LABELED_DATA_FOLDER,
                    str(existing_imgs_count + 1) + ".npy"
                ),
                np.array(self.bboxes)
            )

            print("Annotated image saved to: {}".format(new_img_file))
        else:
            print("No annotation saved for image: {}".format(img_file))

        return

    def keypress(self, event):
        # Undo
        if event.key.lower() == "u":
            assert len(self.bboxes) > 0, "No bboxes to undo"

            self.ax.clear()
            plt.imshow(self.img)

            self.bboxes.pop(-1)
            self.draw_bboxes()

        return

    def onclick(self, event):
        point = [event.xdata, event.ydata]

        bbox = self.point_to_bbox(point)
        self.bboxes.append(bbox)

        self.draw_bboxes()
        return

    def draw_bboxes(self):
        for bbox in self.bboxes:
            # Draw point
            self.ax.scatter(
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
                c='b'
            )

            # Draw bbox lines
            self.ax.plot(
                [bbox[0], bbox[2]],
                [bbox[1], bbox[1]],
                'g'
            )
            self.ax.plot(
                [bbox[0], bbox[2]],
                [bbox[3], bbox[3]],
                'g'
            )
            self.ax.plot(
                [bbox[0], bbox[0]],
                [bbox[1], bbox[3]],
                'g'
            )
            self.ax.plot(
                [bbox[2], bbox[2]],
                [bbox[1], bbox[3]],
                'g'
            )

        self.fig.canvas.draw()
        return

    def point_to_bbox(self, point):
        """
        Convert point into a bounding box with point as center.
        Returns:
            (x_min : int,
            y_min : int,
            x_max : int,
            y_max : int)
        """
        p = [int(point[0]), int(point[1])]

        margin = 15

        return (
            max(0, p[0] - margin), # x_min
            max(0, p[1] - margin), # y_min
            min(self.width, p[0] + margin), # x_max
            min(self.height, p[1] + margin) # y_max
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", action="store_true", default=False,
        help="Flag used to relabel images already in labeled folder")

    args = parser.parse_args()

    # If relabel flag is called, move all images to unlabeled folder
    if args.relabel:
        res = input(
            "Are you sure you want to relabel already labeled images? (y/n)\n"
        )

        if res.lower() == "y":
            labeled_imgs = \
                glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.jpg")) + \
                glob.glob(os.path.join(LABELED_DATA_FOLDER, "*.png"))

            unlabeled_imgs = \
                glob.glob(os.path.join(UNLABELED_DATA_FOLDER, "*.jpg")) + \
                glob.glob(os.path.join(UNLABELED_DATA_FOLDER, "*.png"))

            all_imgs = labeled_imgs + unlabeled_imgs

            os.mkdir(TMP_DATA_FOLDER)

            for i, img in enumerate(all_imgs):
                os.rename(
                    img,
                    os.path.join(
                        TMP_DATA_FOLDER,
                        str(i) + ".png"
                    )
                )

            shutil.rmtree(LABELED_DATA_FOLDER)
            os.mkdir(LABELED_DATA_FOLDER)

            os.rename(TMP_DATA_FOLDER, UNLABELED_DATA_FOLDER)

    img_files = \
        glob.glob(os.path.join(UNLABELED_DATA_FOLDER, "*.jpg")) + \
        glob.glob(os.path.join(UNLABELED_DATA_FOLDER, "*.png"))

    for img_file in img_files:
        Annotator(img_file)







#

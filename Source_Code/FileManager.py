import os
import sys
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class FileManager:

    # Saves an image with a keyword so the image will be called index_keyword.jpg.
    def save_image(self, path, image, keyword, index):
        cv.imwrite(os.path.join(path, str(index) + "_" + str(keyword) + ".jpg"), image)

    # Saves an image without a keyword so the image will be called index.jpg.
    def save_image_nokeyword(self, path, image, index):
        cv.imwrite(os.path.join(path, str(index) + ".jpg"), image)

    # Saves an array with a keyword so the text file will be called index_keyword.txt.
    def save_array(self, path, image, keyword, index):
        with open(os.path.join(path, str(index) + "_" + str(keyword) + ".txt"), 'w') as file:
            file.write(str(image))

    # Opens a text file called "Gamma_Values.txt" and appends value into the file on a new line.
    def save_text_file(self, path, value):
        with open(os.path.join(path, "Gamma_Values.txt"), "a+") as file:
            file.write(str(value) + "\n")
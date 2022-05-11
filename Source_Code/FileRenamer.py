import os
import cv2
from tqdm import tqdm
from FileManager import *


class FileRenamer:
    fileManager = FileManager()

    input_directory = ""
    output_directory = ""

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def rename_files(self, index):
        self.directory_filetype_checker()
        for image in tqdm(os.listdir(self.input_directory)):
            loaded_image = cv2.imread(self.input_directory + "/" + str(image), cv2.IMREAD_COLOR)
            self.fileManager.save_image_nokeyword("Images_After_Rename", loaded_image, index=index)
            index += 1

    def directory_filetype_checker(self):
        for image in tqdm(os.listdir(self.input_directory)):
            if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
                continue
            else:
                raise ValueError("Image is not of filetype 'jpg' or 'png' stopping generator.")

        # If checker verifies integrity of files successfully
        print("Filetype Checker: All images satisfy filetype requirements")


if __name__ == '__main__':
    fileRenamer = FileRenamer("Images_To_Rename", "Images_After_Rename")
    fileRenamer.rename_files(349)

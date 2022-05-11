import os
import cv2
from tqdm import tqdm
from FileManager import *
from DirectoryIntegrityChecker import *


class FileRenamer:
    # Class Instantiation(s)
    fileManager = FileManager()
    direcIntegChecker = DirectoryIntegrityChecker()

    input_directory = ""
    output_directory = ""

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def rename_files(self, first_file_index, verbose):
        if verbose:
            print("FileRenamer: Running function -> 'rename_files' ...")
        
        for image in tqdm(os.listdir(self.input_directory)):
            loaded_image = cv2.imread(self.input_directory + "/" + str(image), cv2.IMREAD_COLOR)
            self.fileManager.save_image_nokeyword("Images_After_Rename", loaded_image, index=first_file_index)
            first_file_index += 1

if __name__ == '__main__':
    fileRenamer = FileRenamer("Images_To_Rename", "Images_After_Rename")
    fileRenamer.rename_files(349)

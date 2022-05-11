import tqdm as tqdm
import os

class DirectoryIntegrityChecker:

    def directory_filetype_checker(self, verbose):
        if verbose:
            print("DirectoryIntegrityChecker: Running function -> 'directory_filetype_checker' ...")
        for image in tqdm(os.listdir(self.input_directory)):
            if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
                continue
            else:
                raise ValueError("Image is not of filetype 'jpg' or 'png'" + "\n" + "Stopping Script...")

        # If checker verifies integrity of files successfully
        print("Filetype Checker: All images satisfy filetype requirements")

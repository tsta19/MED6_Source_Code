import os
import cv2 as cv
from tqdm import tqdm


class Downsampling:
    # folder_dir = "Input_Directory_Portrait"  # image path/directory
    folder_dir_landscape = "Source_Code//DATA//TRAIN_LANDSCAPE"  # image directory
    folder_dir_portrait = "Source_Code//DATA//TRAIN_PORTRAIT"
    folder_dir_both = "Source_Code//DATA//DATA_BOTH"

    resized_img = []
    recolored_img = []
    HSV_img = []

    def import_images(self, verbose: bool):
        if verbose:
            print("Downsampling: Running function -> 'import_images' ...")
        original_img = []
        # print("Importing images...")
        # print("Importing from directory: ", os.listdir(self.folder_dir))
        dir_both = os.listdir(self.folder_dir_both)

        for index in range(0, len(dir_both)):
            if verbose:
                print(f"Imported Image: {dir_both[index]} / {len(dir_both)}", end="\r")
            img = cv.imread(self.folder_dir_both + "//" + dir_both[index])
            if img is not None:
                original_img.append(img)
            else:
                if verbose:
                    print(f"Imported Image Error: {dir_both[index]} / {len(dir_both)}", end="\r")

        if verbose:
            print("Downsampling: Function -> 'import_images' is done")

        return original_img

    def rescale_images(self, verbose):
        if verbose:
            print("Downsampling: Running function -> 'rescale_images' ...")
        original_img = self.import_images(True)
        # new image dimensions
        width = 64
        height = 64
        dimensions = (width, height)

        for img in original_img:
            height = img.shape[0]
            width = img.shape[1]

            # where to chop
            if width > height:
                y1 = int(img.shape[0] / 2 - height / 2)
                y2 = int(img.shape[0] / 2 + height / 2)
                x1 = int(img.shape[1] / 2 - height / 2)
                x2 = int(img.shape[1] / 2 + height / 2)
            else:
                y1 = int(img.shape[0] / 2 - width / 2)
                y2 = int(img.shape[0] / 2 + width / 2)
                x1 = int(img.shape[1] / 2 - width / 2)
                x2 = int(img.shape[1] / 2 + width / 2)

            # crop image
            img = img[y1:y2, x1:x2]

            # resize image
            img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
            self.resized_img.append(img)

        if verbose:
            print("Downsampling: Function -> 'rescale_images' is done")

        return self.resized_img

    def recolor_images(self, verbose):
        if verbose:
            print("Downsampling: Running function -> 'recolor_images' ...")
        for img in self.resized_img:
            height = img.shape[0]
            width = img.shape[1]

            # remove grey pixels!
            for y in range(0, height - 1):  # row
                for x in range(0, width - 1):  # column

                    i = 0  # how far away from the current pixel we look for other pixels
                    kernel = []  # create a list of surrounding pixels

                    # checks if the color channels are similar
                    while all(colors == img[y, x][0] for colors in img[y, x]):

                        i += 1

                        # check if the pixels are inside the image
                        if x > 0 + i and y > 0 + i:
                            kernel.append(img[y - i, x - i])
                        if y > 0 + i:
                            kernel.append(img[y - i, x])
                        if x < width - i and y > 0 + i:
                            kernel.append(img[y - i, x + i])
                        if x > 0 + i:
                            kernel.append(img[y, x - i])
                        kernel.append(img[y, x])
                        if x < width - i:
                            kernel.append(img[y, x + i])
                        if x > 0 + i and y < height - i:
                            kernel.append(img[y + i, x - i])
                        if y < height - i:
                            kernel.append(img[y + i, x])
                        if x < width - i and y < height - i:
                            kernel.append(img[y + i, x + i])

                        # calculate sum of pixels within kernel
                        kernel_sum = [0, 0, 0]
                        for pixel in kernel:
                            kernel_sum += pixel

                        # calculate average and change value of pixel
                        average = kernel_sum / len(kernel)
                        img[y, x] = [round(average[0]), round(average[1]), round(average[2])]

            self.recolored_img.append(img)

        if verbose:
            print("Downsampling: Function -> 'recolor_images' is done")

        return self.recolored_img

    def rescale_image(self, image, res_x, res_y, verbose):
        if verbose:
            print("Downsampling: Running function -> 'rescale_image' ...")
        rescale_dimensions = (res_y, res_x)
        rescaled_image = cv.resize(image, rescale_dimensions, interpolation=cv.INTER_AREA)
        rescaled_image = cv.cvtColor(rescaled_image, cv.COLOR_BGR2RGB)
        if verbose:
            print("Downsampling: Function -> 'rescale_image' is done")
        return rescaled_image

    def BGR2HSV(self, verbose):
        if verbose:
            print("Downsampling: Running function -> 'BGR2HSV' ...")
        for img in self.resized_img:
            # convert to HSV:
            # In OpenCV, Hue has values from 0 to 180, Saturation and Value from 0 to 255.
            # Thus, OpenCV uses HSV ranges between (0-180, 0-255, 0-255)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            self.HSV_img.append(img)
        if verbose:
            print("Downsampling: Function -> 'BGR2HSV' is done")
        return self.HSV_img

    def hsv_images(self):
        self.import_images()
        self.rescale_images()
        self.recolor_images()
        self.BGR2HSV()

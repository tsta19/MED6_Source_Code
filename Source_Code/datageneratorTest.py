import glob
import cv2
import numpy as np
import random
"""
TO ANYONE LOOKING AT THIS CODE:
This was the script to generate the 1000 test data pictures. The script to
generate the 12000 training dataset sadly somehow got lost during a failed Github Push
however it looked a lot like this code, instead it was a nested forloop that for every picture
made 4 new pictures with a random gamma value from the range 0.5-1.5. In this an array holding
the 4 values made sure that a value could not be given twice so we ended up with duplicates.
"""


"""---Begin Variables---"""
testPicPath = "C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/testmappe/originalportraitResized/*.jpg"
testpictures = glob.glob(testPicPath)
img_size = 128
imageArray = []
indexPic = 0
badPics = []
testPics = []
index = 0
indexpicnummer = 0
"""---End Variables---"""

# Reverse gamma function that edits the picture with a value
# that can be used in the normal gamma function to revert the picture
# to its original state.
def adjust_gamma_reverse(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
# The normal gamma function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Append the pictures in the folder to an array.
for i in testpictures:
    indexpicnummer += 1
    image2 = cv2.imread(i)
    testPics.append(image2)
    print(indexpicnummer)

# Gives the 1000 pictures a random value to create bad lit pictures like in the training set
# for the network to predict on.
for i in testPics:
    indexPic += 1
    random_gamma_value = round(random.uniform(0.5,1.5),1)
    gamma_pic = adjust_gamma_reverse(i, gamma=random_gamma_value)
    badPics.append(gamma_pic)
    cv2.imwrite('C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/testmappe/portraitredigeret/' + str(indexPic) + '.jpg', badPics[index])
    with open('C:/Users/krell/OneDrive/Dokumenter/GitHub/P6ContentAwareEditing/KristianNN/testmappe/testporttraitgammavals.txt', 'a') as g:
        g.write(str(random_gamma_value))
        g.write('\n')
    index += 1


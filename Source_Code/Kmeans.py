import glob
import shutil
from sklearn.cluster import KMeans

from FileManager import *


class Kmeans:
    fileManager = FileManager()

    # save images to:
    imdir = "Source_Code//DATA//DATA_BOTH//"
    targetdir = "Source_Code//DATA//SAVE_DIRECTORY"

    filelist = glob.glob(os.path.join(imdir, ('*.jpg' or '*.jpeg')))
    filelist.sort()

    def clustering(self, features, img, verbose: bool):
        if verbose:
            print("Kmeans: Running function -> 'clustering' ...")
        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(np.array(features))

        # Save image to directory
        print("\n")
        for i, clusterGroup in enumerate(kmeans.labels_):
            print("    Copy: %s / %s" % (i, len(kmeans.labels_)), end="\r")
            # fileManager.save_image(targetdir, filelist[i], "cluster_group", str(i))
            if clusterGroup == 0:
                shutil.copy(self.filelist[i],
                            self. targetdir + "//CLUSTER_GROUP_01//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_" + ".jpg")
            elif clusterGroup == 1:
                shutil.copy(self.filelist[i],
                            self.targetdir + "//CLUSTER_GROUP_02//" + str(i) + "_" + "cgroup_" + str(clusterGroup) + "_" + ".jpg")
            else:
                continue
        if verbose:
            print("Kmeans: Function -> 'clustering' is done")

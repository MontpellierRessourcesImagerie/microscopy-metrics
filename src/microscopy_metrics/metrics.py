import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
import math


def signal_to_background_ratio(images, inner_annulus_distance,annulus_thickness):
    """ Measure signal to background ratio of an image.

    Parameters
    ----------
    image : list(np.ndarray)
        The image to be measured.
    inner_annulus_distance : int
        The distance between bead edge and inner annulus edge (in pixels)
    annulus_thickness : int
        The thickness of the annulus.
    Returns
    -------
    signal to background ratio : float
    """
    mean_SBR = 0.0
    SBR = []
    total = 0
    if len(images) == 0 : 
        raise ValueError("You must have at least one PSF")
    for image in images : 
        if image.ndim not in (2,3):
            print("Incorrect picture format")
            pass
        image_float = image.astype(np.float32)
        image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
        threshold_abs = threshold_otsu(image_float)
        image_float[image_float < 0] = 0
        binary_image = image_float > threshold_abs
        signal = 0.0
        n_signal = 0
        background = 0.0
        n_background = 0
        for z in range(binary_image.shape[0]):
            for y in range(binary_image.shape[1]):
                for x in range(binary_image.shape[2]):
                    if binary_image[z][y][x] == 0 :
                        n_background +=1
                        background += image[z][y][x]
                    else :
                        n_signal +=1
                        signal += image[z][y][x]
        ratio = float((signal/n_signal) / (background/n_background))
        SBR.append(ratio)
        mean_SBR += ratio
        total +=1
    return mean_SBR/total,SBR

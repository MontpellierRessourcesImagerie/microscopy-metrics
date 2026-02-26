import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from sklearn.metrics import r2_score
import math
from .utils import *
from matplotlib import pyplot as plt
from .fitting import *


def uncertainty(pcov):
    """ Measure uncertainty of parameters returned by Gaussian fit.

    Parameters
    ----------
    pcov : list
        The covariance matrix between parameters
    Returns
    -------
    perr : np.array
        The list of uncertainty factor for each parameter
    """
    perr = np.sqrt(np.diag(pcov))
    return perr


def determination(params, coords, psf):
    """ Measure determination coefficient of parameters returned by Gaussian fit.

    Parameters
    ----------
    params : list
        The covariance matrix between parameters
    coords : np.array
        The list of coordinates x in the profile
    psf : np.array
        The initial profile
    Returns
    -------
    r_square : float
        The coefficient representing the quality of the fit
    """
    psf_fit = eval_fun(coords,*params)
    r_squared = r2_score(psf,psf_fit)
    return r_squared


class Metrics(object):
    def __init__(self,image=None):
        self.image = image
        self.images = []
        self.ring_inner_distance = 1.0
        self.ring_thickness = 2.0
        self.pixel_size = [1,1,1]

        self.SBR = []
        self.mean_SBR = 0.0


    def set_normalized_image(self,image):
            if image.ndim not in(2,3):
                raise ValueError("Image have to be in 2D or 3D.")
            image_float = image.astype(np.float32)
            image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
            image_float[image_float < 0] = 0
            return image_float


    def signal_to_background_ratio(self):
        """ Parameters
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

        if len(self.images) == 0 : 
            raise ValueError("You must have at least one PSF")
        for image in self.images : 
            if image.ndim not in (2,3):
                print("Incorrect picture format")
                pass
            image_float = self.set_normalized_image(image)
            threshold_abs = legacy_threshold(image_float)
            binary_image = image_float > threshold_abs
            signal = 0.0
            n_signal = 0
            background = 0.0
            n_background = 0
            for z in range(binary_image.shape[0]):
                for y in range(binary_image.shape[1]):
                    for x in range(binary_image.shape[2]):
                        if binary_image[z,y,x] == 0 :
                            n_background +=1
                            background += image[z,y,x]
                        else :
                            n_signal +=1
                            signal += image[z,y,x]
            if n_background == 0 :
                raise ValueError("There are no background pixel detected")
            mean_background = background / n_background
            if n_signal == 0 :
                raise ValueError("There are no signal pixel detected")
            mean_signal = signal / n_signal

            ratio = float(mean_signal / mean_background)
            SBR.append(ratio)
            mean_SBR += ratio
            total +=1
        self.mean_SBR = mean_SBR / total
        self.SBR = SBR


    def signal_to_background_ratio_annulus(self):
        """ Parameters
        ----------
        image : list(np.ndarray)
            The image to be measured.
        inner_annulus_distance : int
            The distance between bead edge and inner annulus edge (in pixels)
        annulus_thickness : int
            The thickness of the annulus.
        physical_pixel : list(float)
            Physical size of a pixel for each axis (z,y,x)
        Returns
        -------
        signal to background ratio : float
        """
        mean_SBR = 0.0
        SBR = []
        total = 0

        if len(self.images) == 0 : 
            raise ValueError("You must have at least one PSF")
        for image in self.images : 
            if image.ndim not in (2,3):
                print("Incorrect picture format")
                pass
            image_float = self.set_normalized_image(image)
            threshold_abs = legacy_threshold(image_float)
            binary_image = image_float > threshold_abs
            labeled_image = label(binary_image)
            regions = regionprops(labeled_image)
            if not regions :
                pass
            largest_region = max(regions,key=lambda r:r.area)
            center = largest_region.centroid
            diameter_bead = np.sqrt(4 * largest_region.area / np.pi)
            inner_distance = um_to_px(self.ring_inner_distance,self.pixel_size[2]) + diameter_bead/2
            outer_distance = um_to_px(self.ring_thickness,self.pixel_size[2]) + inner_distance
            signal = 0.0
            n_signal = 0
            background = 0.0
            n_background = 0
            for z in range(binary_image.shape[0]):
                for y in range(binary_image.shape[1]):
                    for x in range(binary_image.shape[2]):
                        distance = dist(center,[y,x])
                        if binary_image[z][y][x] == 0 :
                            if distance>=inner_distance and distance<=outer_distance:
                                n_background +=1
                                background += image[z][y][x]
                        else :
                            n_signal +=1
                            signal += image[z][y][x]
            if n_background == 0 :
                raise ValueError("There are no background pixel detected")
            mean_background = background / n_background
            if n_signal == 0 :
                raise ValueError("There are no signal pixel detected")
            mean_signal = signal / n_signal
            ratio = float(mean_signal / mean_background)
            SBR.append(ratio)
            mean_SBR += ratio
            total +=1
        self.mean_SBR = mean_SBR / total
        self.SBR = SBR


    def run_prefitting_metrics(self):
        self.SBR = []
        self.mean_SBR = 0.0
        yield {'desc':"SBR calculation..."}
        self.signal_to_background_ratio_annulus()
    


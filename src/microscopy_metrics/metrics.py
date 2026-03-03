import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from sklearn.metrics import r2_score
from scipy.ndimage import median_filter
import math
from .utils import *
from matplotlib import pyplot as plt
from .fitting import *

class Metrics(object):
    def __init__(self,image=None):
        self.image = image
        self.images = []
        self.ring_inner_distance = 1.0
        self.ring_thickness = 2.0
        self.pixel_size = [1,1,1]
        self.FWHM = []
        self.LAR = 0
        self.spherict = 0

        self.SBR = []
        self.mean_SBR = 0.0


    def set_normalized_image(self,image):
            if image.ndim not in(2,3):
                raise ValueError("Image have to be in 2D or 3D.")
            image_float = image.astype(np.float32)
            image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
            image_float[image_float < 0] = 0
            return image_float

    def process_single_SBR_ring(self,index,image):
        if image.ndim not in (2,3):
            print("Incorrect picture format")
            return -1
        image_float = self.set_normalized_image(image)
        image_float = median_filter(image_float,size=5)
        threshold_abs = legacy_threshold(image_float,1000)
        binary_image = image_float > threshold_abs
        labeled_image = label(binary_image)
        regions = regionprops(labeled_image)
        if not regions :
            return -1
        largest_region = max(regions,key=lambda r:r.area)
        min_z, min_y, min_x, max_z, max_y, max_x = largest_region.bbox
        center = largest_region.centroid
        diameter_z = max_z - min_z
        diameter_y = max_y - min_y
        diameter_x = max_x - min_x
        diameter_bead = max(diameter_z, diameter_y, diameter_x)  # Prend la dimension maximale
        inner_distance = um_to_px(self.ring_inner_distance,self.pixel_size[2]) + diameter_bead/2
        outer_distance = um_to_px(self.ring_thickness,self.pixel_size[2]) + inner_distance
        signal = 0.0
        n_signal = 0
        background = 0.0
        n_background = 0
        for z in range(binary_image.shape[0]):
            for y in range(binary_image.shape[1]):
                for x in range(binary_image.shape[2]):
                    distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                    if binary_image[z, y, x] == 1:
                        n_signal += 1
                        signal += image[z, y, x]
                    else:
                        if inner_distance <= distance <= outer_distance:
                            n_background += 1
                            background += image[z, y, x]

        if n_background == 0 :
            raise ValueError("There are no background pixel detected")
        mean_background = background / n_background
        if n_signal == 0 :
            raise ValueError("There are no signal pixel detected")
        mean_signal = signal / n_signal
        ratio = float(mean_signal / mean_background)
        return ratio

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

        if len(self.images) == 0 : 
            raise ValueError("You must have at least one PSF")
        total = len(self.images)
        with ThreadPoolExecutor() as executor : 
            futures = {
                executor.submit(lambda i, img: self.process_single_SBR_ring(i, img), i, image): i
                for i, image in enumerate(self.images)
            }

            for future in as_completed(futures):
                result = future.result()
                if result == -1 :
                    total += result
                else :
                    SBR.append(result)
                    mean_SBR += result
        self.mean_SBR = mean_SBR / total
        self.SBR = SBR


    def run_prefitting_metrics(self):
        self.SBR = []
        self.mean_SBR = 0.0
        yield {'desc':"SBR calculation..."}
        self.signal_to_background_ratio_annulus()

    def lateral_asymmetry_ratio(self):
        if self.FWHM == []:
            return
        tmp = np.array([self.FWHM[1], self.FWHM[2]])
        self.LAR = tmp.min()/tmp.max()

    def sphericity(self):
        if self.FWHM == []:
            return 
        FWHM_xy = math.sqrt(self.FWHM[2] * self.FWHM[1])
        sphericity = FWHM_xy / self.FWHM[0]
        self.spherict = sphericity
    


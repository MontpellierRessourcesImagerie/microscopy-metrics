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
from .theoretical_resolution import *
from .threshold_tool import Threshold_Legacy
        

class Metrics(object):
    def __init__(self, image=None):
        self._image = image
        self._images = []
        self._ring_inner_distance = 1.0
        self._ring_thickness = 2.0
        self._pixel_size = [1, 1, 1]
        self._FWHM = []
        self._Theoretical_Resolution_Tool = None

        self.LAR = 0
        self._sphericity = 0
        self.SBR = []
        self.mean_SBR = 0.0
        self.theoretical_resolution = []

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("Please, select an Image with 2 or 3 dimensions.")
        self._image = image

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if len(images) == 0 or images is None:
            raise ValueError("Please, send at list one image")
        self._images = images

    @property
    def ring_inner_distance(self):
        return self._ring_inner_distance

    @ring_inner_distance.setter
    def ring_inner_distance(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ring_inner_distance")
        self._ring_inner_distance = value

    @property
    def ring_thickness(self):
        return self._ring_thickness

    @ring_thickness.setter
    def ring_thickness(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ring_inner_distance")
        self._ring_thickness = value

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Shape format not compatible with current image")
        self._pixel_size = value

    @property
    def FWHM(self):
        return self._FWHM

    @FWHM.setter
    def FWHM(self, value):
        if not isinstance(value, list):
            raise ValueError("FWHM must be a list")
        self._FWHM = value

    @property
    def sphericity(self):
        return self._sphericity

    @sphericity.setter
    def sphericity(self, value):
        if not isinstance(value, float):
            raise ValueError("Sphericity muse be a float")
        self._sphericity = value

    @property
    def theoretical_resolution_tool(self):
        return self._Theoretical_Resolution_Tool

    @theoretical_resolution_tool.setter
    def theoretical_resolution_tool(self, value):
        self._Theoretical_Resolution_Tool = value

    def set_normalized_image(self, image):
        """Method to normalize a 2D or 3D image and erase negative values

        Args:
            image (np.ndarray): Image to be normalized

        Raises:
            ValueError: This function only operate on 2D or 3D images

        Returns:
            np.ndarray: Image normalized
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        image_float = image.astype(np.float32)
        image_float = (image_float - np.min(image_float)) / (
            np.max(image_float) - np.min(image_float) + 1e-6
        )
        image_float[image_float < 0] = 0
        return image_float

    def process_single_SBR_ring(self, index, image):
        """Function to calculate Signa to background ratio using a ring for a specific bead

        Args:
            index (int): Bead ID corresping to it's position in the list
            image (np.ndarray): Image to use for the calculation

        Raises:
            ValueError: There have to be at least one background pixel in the image
            ValueError: There have to be at least one signal pixel in the image

        Returns:
            float: Signal to background ratio of the image
        """
        if image.ndim not in (2, 3):
            print("Incorrect picture format")
            return -1
        image_float = self.set_normalized_image(image)
        image_float = median_filter(image_float, size=5)
        threshold_abs = Threshold_Legacy(nb_iteration=1000).get_threshold(image_float)
        binary_image = image_float > threshold_abs
        labeled_image = label(binary_image)
        regions = regionprops(labeled_image)
        if not regions:
            return -1
        largest_region = max(regions, key=lambda r: r.area)
        min_z, min_y, min_x, max_z, max_y, max_x = largest_region.bbox
        center = largest_region.centroid
        diameter_z = max_z - min_z
        diameter_y = max_y - min_y
        diameter_x = max_x - min_x
        diameter_bead = max(diameter_z, diameter_y, diameter_x)
        inner_distance = (
            um_to_px(self._ring_inner_distance, self._pixel_size[2]) + diameter_bead / 2
        )
        outer_distance = (
            um_to_px(self._ring_thickness, self._pixel_size[2]) + inner_distance
        )
        signal = 0.0
        n_signal = 0
        background = 0.0
        n_background = 0
        for z in range(binary_image.shape[0]):
            for y in range(binary_image.shape[1]):
                for x in range(binary_image.shape[2]):
                    distance = np.sqrt(
                        (z - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (x - center[2]) ** 2
                    )
                    if binary_image[z, y, x] == 1:
                        n_signal += 1
                        signal += image[z, y, x]
                    else:
                        if inner_distance <= distance <= outer_distance:
                            n_background += 1
                            background += image[z, y, x]

        if n_background == 0:
            raise ValueError("There are no background pixel detected")
        mean_background = background / n_background
        if n_signal == 0:
            raise ValueError("There are no signal pixel detected")
        mean_signal = signal / n_signal
        ratio = float(mean_signal / mean_background)
        return ratio

    def signal_to_background_ratio_annulus(self):
        mean_SBR = 0.0
        SBR = []

        if len(self._images) == 0:
            raise ValueError("You must have at least one PSF")
        total = len(self._images)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    lambda i, img: self.process_single_SBR_ring(i, img), i, image
                ): i
                for i, image in enumerate(self._images)
            }

            for future in as_completed(futures):
                result = future.result()
                if result == -1:
                    total += result
                else:
                    SBR.append(result)
                    mean_SBR += result
        self.mean_SBR = mean_SBR / total
        self.SBR = SBR

    def run_prefitting_metrics(self):
        self.SBR = []
        self.mean_SBR = 0.0
        yield {"desc": "SBR calculation..."}
        self.signal_to_background_ratio_annulus()
        yield {"desc": "Estimating theoretical resolution..."}
        self.theoretical_resolution = (
            self._Theoretical_Resolution_Tool.get_theoretical_resolution()
        )

    def lateral_asymmetry_ratio(self):
        if self._FWHM == []:
            return
        tmp = np.array([self._FWHM[1], self._FWHM[2]])
        self.LAR = tmp.min() / tmp.max()

    def sphericity_ratio(self):
        if self._FWHM == []:
            return
        FWHM_xy = math.sqrt(self._FWHM[2] * self._FWHM[1])
        sphericity = FWHM_xy / self._FWHM[0]
        self._sphericity = sphericity

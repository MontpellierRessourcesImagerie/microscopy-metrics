import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu,threshold_isodata,threshold_li,threshold_minimum,threshold_triangle
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from .utils import *
import math
import os
from PIL import Image
from skimage.draw import polygon_perimeter


class Detection(object) :
    """ Standard class for operations relative to detection and extraction of PSFs"""
    def __init__(self, image=None):
        self._image = image
        self._crop_factor = 5
        self._sigma = 3
        self._min_distance = 1
        self._bead_size = 0.6
        self._rejection_distance = 0.5
        self._pixel_size = [0.06,0.06,0.5]
        self._threshold_rel = 0.3
        self.threshold_choice = "otsu"

        self.normalized_image = None
        self.high_passed_im = None
        self.threshold = None
        self.centroids = []
        self.rois_extracted = []
        self.list_id_centroids_retained = []
        self.detect_methods_list = [
            self.detect_psf_peak_local_max, 
            self.detect_psf_blob_log, 
            self.detect_psf_blob_dog, 
            self.detect_psf_centroid
        ]
        self.cropped = []

    @property
    def image(self):
        return self._image
    @image.setter
    def image(self,image):
        if not isinstance(image,np.ndarray) or image.ndim not in (2,3):
            raise ValueError("Please, select an Image with 2 or 3 dimensions.")
        self._image = image
    
    @property
    def crop_factor(self):
        return self._crop_factor
    @crop_factor.setter
    def crop_factor(self,value):
        if not isinstance(value,int) or value > np.max(self._image.shape):
            raise ValueError("Please, choose an integer smaller than image as a crop factor")
        self._crop_factor = value

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self,value):
        if not isinstance(value,int):
            raise ValueError("Please, choose an integer as a sigma")
        self._sigma = value
    
    @property
    def min_distance(self):
        return self._min_distance
    @min_distance.setter
    def min_distance(self,value):
        if not isinstance(value,int) :
            raise ValueError("Please, choose an integer as a minimal distance")
        self._min_distance = value
    
    @property
    def bead_size(self):
        return self._bead_size
    @bead_size.setter
    def bead_size(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a size of bead")
        self._bead_size = value
    
    @property
    def rejection_distance(self):
        return self._rejection_distance
    @rejection_distance.setter
    def rejection_distance(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a rejection distance")
        self._rejection_distance = value
    
    @property
    def pixel_size(self):
        return self._pixel_size
    @pixel_size.setter
    def pixel_size(self,value):
        if not isinstance(value,np.ndarray):
            raise ValueError("Shape format not compatible with current image")
        self._pixel_size = value
    
    @property
    def threshold_rel(self):
        return self._threshold_rel
    @threshold_rel.setter
    def threshold_rel(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a threshold")
        self._threshold_rel = value
    

    def set_threshold(self):
        self.threshold = self._threshold_rel * np.max(self.high_passed_im)
        if self.threshold_choice is not None : 
            if self.threshold_choice  == "isodata":
                self.threshold = threshold_isodata(self.high_passed_im)
            elif self.threshold_choice  == "li":
                self.threshold = threshold_li(self.high_passed_im)
            elif self.threshold_choice  == "minimum":
                self.threshold = threshold_minimum(self.high_passed_im)
            elif self.threshold_choice  == "triangle":
                self.threshold = threshold_triangle(self.high_passed_im)
            else :
                self.threshold = threshold_otsu(self.high_passed_im)


    def set_normalized_image(self):
        """Method to normalize a 2D or 3D image and erase negative values

        Raises:
            ValueError: This function only operate on 2D or 3D images
        """
        if self._image.ndim not in(2,3):
            raise ValueError("Image have to be in 2D or 3D.")
        self.normalized_image = self._image.astype(np.float32)
        self.normalized_image = (self.normalized_image - np.min(self.normalized_image)) / (np.max(self.normalized_image) - np.min(self.normalized_image) + 1e-6)
        self.normalized_image[self.normalized_image < 0] = 0


    def gaussian_high_pass(self):
        low_pass = ndi.gaussian_filter(self.normalized_image,self._sigma)
        self.high_passed_im = self.normalized_image - low_pass
        self.set_threshold()


    def detect_psf_peak_local_max(self):
        self.set_normalized_image()
        self.normalized_image = ndi.gaussian_filter(self.normalized_image,sigma=2.0)
        self.gaussian_high_pass()
        self.centroids = peak_local_max(self.high_passed_im,min_distance=self._min_distance,threshold_abs=self.threshold)


    def detect_psf_blob_log(self):
        self.set_normalized_image()
        self.high_passed_im = self.normalized_image
        self.set_threshold()
        blobs = blob_log(self.normalized_image, max_sigma=self._sigma, threshold=self.threshold)
        self.centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])


    def detect_psf_blob_dog(self):
        self.set_normalized_image()
        self.high_passed_im = self.normalized_image
        self.set_threshold()
        blobs = blob_dog(self.normalized_image, max_sigma=self._sigma, threshold=self.threshold)
        self.centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])


    def detect_psf_centroid(self):
        self.set_normalized_image()
        self.normalized_image = ndi.gaussian_filter(self.normalized_image,sigma=2.0)
        self.gaussian_high_pass()
        binary_image = self.high_passed_im > self.threshold
        labeled_image = label(binary_image)
        region_props = regionprops(labeled_image)
        tmp_centroids = []
        for prop in region_props :
            tmp_centroids.append(prop.centroid)

        self.centroids = np.array(tmp_centroids)


    def extract_Region_Of_Interest(self):
        """ Uses found centroids to extract region of interest 
        Automatically rejects the ones overlapped or too near from the edges.
        """
        roi_size = um_to_px((self._crop_factor * self._bead_size) / 2, self._pixel_size[2])
        for i,centroid in enumerate(self.centroids) :
            over = False
            for y,c2 in enumerate(self.centroids) : 
                if i != y :
                    if math.dist(centroid,c2)<(math.sqrt(2) * roi_size):
                        over = True
                        break
            if over == False : 
                tmp = np.array([
                        [int(centroid[0]),math.ceil(centroid[1] - roi_size), math.ceil(centroid[2] - roi_size)],
                        [int(centroid[0]),math.ceil(centroid[1] - roi_size), math.ceil(centroid[2] + roi_size)],
                        [int(centroid[0]),math.ceil(centroid[1] + roi_size), math.ceil(centroid[2] + roi_size)],
                        [int(centroid[0]),math.ceil(centroid[1] + roi_size), math.ceil(centroid[2] - roi_size)],           
                    ]
                )
                overlapped = is_roi_overlapped(self.rois_extracted,tmp)
                if not overlapped :
                    if is_roi_in_image(tmp, self._image.shape) and is_roi_not_in_rejection(centroid,self._image.shape,math.ceil(um_to_px(self._rejection_distance,self._pixel_size[0]))):
                        self.rois_extracted.append(tmp)
                        self.list_id_centroids_retained.append(i)


    def run(self, selected_tool, output_dir=None, crop_psf=True):
        """Function to operate complete detection workflow

        Args:
            selected_tool (int): Selected detection tool by the user
            output_dir (Path, optional): Directory of the output folder. Defaults to None.
            crop_psf (bool, optional): Allow or not the generation of cropped PSF images. Defaults to True.

        Raises:
            ValueError: To generate images of the cropped PSFs, the output_dir have to exist

        Yields:
            String: Return the current step of the workflow
        """
        if output_dir is None and crop_psf == True :
            raise ValueError("Problem to find output folder")
        self.centroids = []
        self.rois_extracted = []
        self.list_id_centroids_retained = []
        self.cropped = []
        self.detect_methods_list[selected_tool]()
        yield {'desc':"Extracting Rois..."}
        self.extract_Region_Of_Interest()
        if crop_psf:
            yield {'desc': "Cropping PSFs..."}
            self.crop_psf(output_dir)


    def get_active_path(self, index,output_dir):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list
            output_dir (Path): Directory of the output folder

        Returns:
            Path: Folder's path found (or created) for the selected bead 
        """
        active_path = os.path.join(output_dir,f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path


    def add_roi_on_image(self,roi):
        """Function to draw a square representating an ROI in a picture

        Args:
            roi (np.array): List of the four corners coordinates of the ROI

        Returns:
            np.ndarray: Modified image with the ROI
        """
        if self._image.ndim == 3 :
            image_tmp = np.max(self._image,axis=0)
        image_tmp = ((image_tmp-image_tmp.min()) / (image_tmp.max() - image_tmp.min()) * 255).astype(np.uint8)
        image_rgb = np.stack([image_tmp,image_tmp,image_tmp], axis=-1)
        rr, cc = polygon_perimeter(
            [roi[0, 1], roi[1, 1], roi[2, 1], roi[3, 1]],
            [roi[0, 2], roi[1, 2], roi[2, 2], roi[3, 2]],
            image_tmp.shape
        )
        image_rgb[rr,cc,0] = 255
        image_rgb[rr,cc,1] = 255
        image_rgb[rr,cc,2] = 255
        return image_rgb


    def crop_psf(self, output_dir):
        """Function to crop image for each ROI and save them

        Args:
            output_dir (Path): Directory of the output folder
        """
        for i, roi in enumerate(self.rois_extracted):
            data = self._image[...,roi[0][1]:roi[2][1],roi[0][2]:roi[1][2]]
            self.cropped.append(data)
            active_path = self.get_active_path(i,output_dir)
            centroid_idx = self.list_id_centroids_retained[i]
            physic = [int(self.centroids[centroid_idx][0]),int(self.centroids[centroid_idx][1] - self.rois_extracted[i][0][1]),int(self.centroids[centroid_idx][2] - self.rois_extracted[i][0][2])]
            image_float = data.astype(np.float32)
            image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
            image_float[image_float < 0] = 0
            image_uint16 = (image_float * 255).astype(np.uint8)
            XY_data = Image.fromarray(image_uint16[physic[0],:,:])
            YZ_data = Image.fromarray(image_uint16[:,:,physic[2]])
            XZ_data = Image.fromarray(image_uint16[:,physic[1],:])
            XY_data.save(os.path.join(active_path,"XY_view.png"))
            YZ_data.save(os.path.join(active_path,"YZ_view.png"))
            XZ_data.save(os.path.join(active_path,"XZ_view.png"))
            image_roi = Image.fromarray(self.add_roi_on_image(roi))
            image_roi.save(os.path.join(active_path,"Localisation.png"))
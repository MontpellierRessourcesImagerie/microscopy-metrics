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
    def __init__(self, image=None):
        self.image = image
        self.crop_factor = 5
        self.sigma = 3
        self.min_distance = 1
        self.bead_size = 0.6
        self.rejection_distance = 0.5
        self.pixel_size = [0.06,0.06,0.5]
        self.threshold_rel = 0.3
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


    def set_image(self,image):
        if not isinstance(image,np.ndarray) or image.ndim not in (2,3):
            raise ValueError("Please, select an Image with 2 or 3 dimensions.")
        self.image = image
    

    def get_image(self):
        return self.image


    def set_crop_factor(self,value):
        if not isinstance(value,int) or value > np.max(self.image.shape):
            raise ValueError("Please, choose an integer smaller than image as a crop factor")
        self.crop_factor = value
    

    def get_crop_factor(self):
        return self.crop_factor


    def set_sigma(self,value):
        if not isinstance(value,int):
            raise ValueError("Please, choose an integer as a sigma")
        self.sigma = value
    

    def get_sigma(self):
        return self.sigma


    def set_min_distance(self,value):
        if not isinstance(value,int) :
            raise ValueError("Please, choose an integer as a minimal distance")
        self.min_distance = value
    

    def get_min_distance(self):
        return self.min_distance


    def set_bead_size(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a size of bead")
        self.bead_size = value
    

    def get_bead_size(self):
        return self.bead_size


    def set_rejection_distance(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a rejection distance")
        self.rejection_distance = value
    

    def get_rejection_distance(self):
        return self.rejection_distance


    def set_pixel_size(self,value):
        if not isinstance(value,np.array) or value.shape != self.image.shape:
            raise ValueError("Shape format not compatible with current image")
        self.pixel_size = value
    

    def get_pixel_size(self):
        return self.pixel_size


    def set_bead_size(self,value):
        if not isinstance(value,float) :
            raise ValueError("Please, choose a float as a size of bead")
        self.bead_size = value
    

    def get_bead_size(self):
        return self.bead_size


    def set_threshold(self):
        self.threshold = self.threshold_rel * np.max(self.high_passed_im)
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
        if self.image.ndim not in(2,3):
            raise ValueError("Image have to be in 2D or 3D.")
        self.normalized_image = self.image.astype(np.float32)
        self.normalized_image = (self.normalized_image - np.min(self.normalized_image)) / (np.max(self.normalized_image) - np.min(self.normalized_image) + 1e-6)
        self.normalized_image[self.normalized_image < 0] = 0


    def gaussian_high_pass(self):
        """ Apply a gaussian high pass filter to an image.

        Parameters
        ----------
        image : np.ndarray
            The image to be filtered.
        sigma : float
            The sigma (width) of the gaussian filter to be applied.
            The default value is 2.

        Returns
        -------
        high_passed_im : np.ndarray
            The image with the high pass filter applied
        """
        low_pass = ndi.gaussian_filter(self.normalized_image,self.sigma)
        self.high_passed_im = self.normalized_image - low_pass
        self.set_threshold()


    def detect_psf_peak_local_max(self):
        """ Detect coords of PSFs in an image.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed.
        min_distance : int
            Minimal distance between two PSFs (pixels)
        threshold_rel : float
            Relative threshold
        

        Returns
        -------
        Coordinates : np.ndarray
            List of coordinates (y,x) of PSFs
        """
        self.set_normalized_image()
        self.normalized_image = ndi.gaussian_filter(self.normalized_image,sigma=2.0)
        self.gaussian_high_pass()
        self.centroids = peak_local_max(self.high_passed_im,min_distance=self.min_distance,threshold_abs=self.threshold)


    def detect_psf_blob_log(self):
        """ Detect coords of PSF's centroids using blob log.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed.
        min_sigma : float
            Minimal size of PSFs (pixels).
        max_sigma : float
            Maximal size of PSFs (pixels)
        threshold_rel : float
            Relative threshold
        
        Returns
        -------
        Coordinates : np.ndarray
            List of coordinates (y,x) of PSFs
        """
        self.set_normalized_image()
        self.high_passed_im = self.normalized_image
        self.set_threshold()
        blobs = blob_log(self.normalized_image, max_sigma=self.sigma, threshold=self.threshold)
        self.centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])


    def detect_psf_blob_dog(self):
        """ Detect coords of PSF's centroids using blob dog.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed.
        min_sigma : float
            Minimal size of PSFs (pixels).
        max_sigma : float
            Maximal size of PSFs (pixels)
        threshold_rel : float
            Relative threshold
        

        Returns
        -------
        Coordinates : np.ndarray
            List of coordinates (y,x) of PSFs
        """
        self.set_normalized_image()
        self.high_passed_im = self.normalized_image
        self.set_threshold()
        blobs = blob_dog(self.normalized_image, max_sigma=self.sigma, threshold=self.threshold)
        self.centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])


    def detect_psf_centroid(self):
        """ Detect coords of PSF's centroids.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed.
        threshold_rel : float
            Relative threshold
        

        Returns
        -------
        Coordinates : np.ndarray
            List of coordinates (y,x) of PSFs
        """
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
        """ Uses centroids of detected beads to extract region of interest for each and remove the ones overlapped or too near from the edges.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed
        centroids : np.ndarray
            The list of coordinates of each centroid
        crop_factor : integer
            The size factor of the ROI
        bead_size : float
            The theoretical size of a bead
        rejection_zone : float
            Minimal distance between bead and Top/Bottom
        physical_pixel : list[float]
            Physical dimensions of a pixel (um/px)
        
        Returns
        -------
        Shapes : List of np.ndarray
            List of all shapes representing ROIs
        """
        roi_size = um_to_px((self.crop_factor * self.bead_size) / 2, self.pixel_size[2])
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
                    if is_roi_in_image(tmp, self.image.shape) and is_roi_not_in_rejection(centroid,self.image.shape,math.ceil(um_to_px(self.rejection_distance,self.pixel_size[0]))):
                        self.rois_extracted.append(tmp)
                        self.list_id_centroids_retained.append(i)


    def run(self, selected_tool, output_dir=None):
        if output_dir is None :
            raise ValueError("Problem to find output folder")
        self.centroids = []
        self.rois_extracted = []
        self.list_id_centroids_retained = []
        self.detect_methods_list[selected_tool]()
        yield {'desc':"Extracting Rois..."}
        self.extract_Region_Of_Interest()
        yield {'desc': "Cropping PSFs..."}
        self.on_crop_psf(output_dir)


    def get_active_path(self, index,output_dir):
        """Utility function to return the current path of a given bead"""
        active_path = os.path.join(output_dir,f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path


    def add_roi_on_image(self,roi):
        if self.image.ndim == 3 :
            image_tmp = np.max(self.image,axis=0)
        image_tmp = ((image_tmp-image_tmp.min()) / (image_tmp.max() - image_tmp.min()) * 255).astype(np.uint8)
        image_rgb = np.stack([image_tmp,image_tmp,image_tmp], axis=-1)
        rr, cc = polygon_perimeter(
            [roi[0, 1], roi[1, 1], roi[2, 1], roi[3, 1]],
            [roi[0, 2], roi[1, 2], roi[2, 2], roi[3, 2]],
            image_tmp.shape
        )
        image_rgb[rr,cc,0] = 255
        image_rgb[rr,cc,1] = 0
        image_rgb[rr,cc,2] = 0
        return image_rgb


    def on_crop_psf(self, output_dir):
        for i, roi in enumerate(self.rois_extracted):
            data = self.image[...,roi[0][1]:roi[2][1],roi[0][2]:roi[1][2]]
            self.cropped.append(data)
            active_path = self.get_active_path(i,output_dir)
            centroid_idx = self.list_id_centroids_retained[i]
            physic = [int(self.centroids[centroid_idx][0]),int(self.centroids[centroid_idx][1] - self.rois_extracted[i][0][1]),int(self.centroids[centroid_idx][2] - self.rois_extracted[i][0][2])]
            image_float = data.astype(np.float32)
            image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
            image_float[image_float < 0] = 0
            image_uint16 = (image_float * 65535).astype(np.uint16)
            XY_data = Image.fromarray(image_uint16[physic[0],:,:])
            YZ_data = Image.fromarray(image_uint16[:,:,physic[2]])
            XZ_data = Image.fromarray(image_uint16[:,physic[1],:])
            XY_data.save(os.path.join(active_path,"XY_view.png"))
            YZ_data.save(os.path.join(active_path,"YZ_view.png"))
            XZ_data.save(os.path.join(active_path,"XZ_view.png"))
            image_roi = Image.fromarray(self.add_roi_on_image(roi))
            image_roi.save(os.path.join(active_path,"Localisation.png"))
import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu,threshold_isodata,threshold_li,threshold_minimum,threshold_triangle
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
from .utils import *
import math

def gaussian_high_pass(image:np.ndarray, sigma:float = 2):
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
    low_pass = ndi.gaussian_filter(image,sigma)
    high_passed_im = image - low_pass

    return high_passed_im


def detect_psf_peak_local_max(image, min_distance=5, threshold_rel=0.3, threshold_auto=False, threshold_choice="otsu"):
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
    if image.ndim not in(2,3):
        raise ValueError("Image have to be in 2D or 3D.")
    image_float = image.astype(np.float32)
    image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
    rescaled_image = ndi.gaussian_filter(image_float,sigma=2.0)
    filtered_image = gaussian_high_pass(rescaled_image, sigma = 2.0)
    threshold_abs = threshold_rel * np.max(filtered_image)
    if threshold_auto : 
        if threshold_choice == "isodata":
            threshold_abs = threshold_isodata(filtered_image)
        elif threshold_choice == "li":
            threshold_abs = threshold_li(filtered_image)
        elif threshold_choice == "minimum":
            threshold_abs = threshold_minimum(filtered_image)
        elif threshold_choice == "triangle":
            threshold_abs = threshold_triangle(filtered_image)
        else :
            threshold_abs = threshold_otsu(filtered_image)
    coordinates = peak_local_max(filtered_image,min_distance=min_distance,threshold_abs=threshold_abs)
    return coordinates

def detect_psf_blob_log(image, max_sigma = 3, threshold_rel=0.1, threshold_auto=False, threshold_choice="otsu"):
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
    if image.ndim not in (2,3):
        raise ValueError("Image have to be in 2D or 3D.")
    image_float = image.astype(np.float32)
    image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
    
    threshold_abs = threshold_rel * np.max(image_float)
    if threshold_auto :
        if threshold_choice == "isodata":
            threshold_abs = threshold_isodata(image_float)
        elif threshold_choice == "li":
            threshold_abs = threshold_li(image_float)
        elif threshold_choice == "minimum":
            threshold_abs = threshold_minimum(image_float)
        elif threshold_choice == "triangle":
            threshold_abs = threshold_triangle(image_float)
        else :
            threshold_abs = threshold_otsu(image_float)
    filtered_image = gaussian_high_pass(image_float, sigma = max_sigma)
    image_float[image_float < 0] = 0
    
    centroids = []
    blobs = blob_log(image_float, min_sigma=1, max_sigma=max_sigma, threshold=threshold_abs)
    centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])
    return centroids

def detect_psf_blob_dog(image, max_sigma = 3, threshold_rel=0.1, threshold_auto=False, threshold_choice="otsu"):
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
    if image.ndim not in (2,3):
        raise ValueError("Image have to be in 2D or 3D.")
    image_float = image.astype(np.float32)
    image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
    threshold_abs = threshold_rel * np.max(image_float)
    if threshold_auto :
        if threshold_choice == "isodata":
            threshold_abs = threshold_isodata(image_float)
        elif threshold_choice == "li":
            threshold_abs = threshold_li(image_float)
        elif threshold_choice == "minimum":
            threshold_abs = threshold_minimum(image_float)
        elif threshold_choice == "triangle":
            threshold_abs = threshold_triangle(image_float)
        else :
            threshold_abs = threshold_otsu(image_float)
    filtered_image = gaussian_high_pass(image_float, sigma = max_sigma)
    image_float[image_float < 0] = 0
    centroids = []
    blobs = blob_dog(image_float, min_sigma=1, max_sigma=max_sigma, threshold=threshold_abs)
    centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])
    return centroids

def detect_psf_centroid(image, threshold_rel=0.1, threshold_auto=False, threshold_choice="otsu"):
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
    if image.ndim not in(2,3):
        raise ValueError("Image have to be in 2D or 3D.")
    image_float = image.astype(np.float32)
    image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)
    rescaled_image = ndi.gaussian_filter(image_float,sigma=2.0)
    filtered_image = gaussian_high_pass(rescaled_image, sigma = 2.0)
    threshold_abs = threshold_rel * np.max(filtered_image)
    if threshold_auto :
        if threshold_choice == "isodata":
            threshold_abs = threshold_isodata(filtered_image)
        elif threshold_choice == "li":
            threshold_abs = threshold_li(filtered_image)
        elif threshold_choice == "minimum":
            threshold_abs = threshold_minimum(filtered_image)
        elif threshold_choice == "triangle":
            threshold_abs = threshold_triangle(filtered_image)
        else :
            threshold_abs = threshold_otsu(filtered_image)
    binary_image = filtered_image > threshold_abs
    labeled_image = label(binary_image)
    centroids = regionprops(labeled_image)
    ret = []
    for centroid in centroids :
        ret.append(centroid.centroid)

    retf = np.array(ret)
    return retf

def extract_Region_Of_Interest(image,centroids, crop_factor=5, bead_size=10, rejection_zone=10, physical_pixel=[1,1,1]):
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
    rois = []
    centroids_retained = []
    image_shape = get_shape(image)
    roi_size = um_to_px((crop_factor * bead_size) / 2, physical_pixel[2])
    for i,centroid in enumerate(centroids) :
        over = False
        for y,c2 in enumerate(centroids) : 
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
            overlapped = is_roi_overlapped(rois,tmp)
            if not overlapped :
                if is_roi_in_image(tmp, image_shape) and is_roi_not_in_rejection(centroid,image_shape,math.ceil(um_to_px(rejection_zone,physical_pixel[0]))):
                    rois.append(tmp)
                    centroids_retained.append(i)
    return rois,centroids_retained


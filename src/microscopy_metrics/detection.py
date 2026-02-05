import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
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


def detect_psf_peak_local_max(image, min_distance=5, threshold_rel=0.3, threshold_auto=False):
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
    rescaled_image = adjust_sigmoid(image_float)
    filtered_image = gaussian_high_pass(rescaled_image, sigma = 10)
    threshold_abs = threshold_rel * np.max(filtered_image)
    if threshold_auto : 
        threshold_abs = threshold_otsu(filtered_image)
    coordinates = peak_local_max(filtered_image,min_distance=min_distance,threshold_abs=threshold_abs)
    return coordinates

def detect_psf_blob_log(image, max_sigma = 3, threshold_rel=0.1, threshold_auto=False):
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
        threshold_abs = threshold_otsu(image_float)
    filtered_image = gaussian_high_pass(image_float, sigma = max_sigma)
    image_float[image_float < 0] = 0
    
    centroids = []
    blobs = blob_log(image_float, min_sigma=1, max_sigma=max_sigma, threshold=threshold_abs)
    centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])
    return centroids

def detect_psf_blob_dog(image, max_sigma = 3, threshold_rel=0.1, threshold_auto=False):
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
        threshold_abs = threshold_otsu(image_float)
    filtered_image = gaussian_high_pass(image_float, sigma = max_sigma)
    image_float[image_float < 0] = 0
    centroids = []
    blobs = blob_dog(image_float, min_sigma=1, max_sigma=max_sigma, threshold=threshold_abs)
    centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])
    return centroids

def detect_psf_centroid(image, threshold_rel=0.1, threshold_auto=False):
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
    rescaled_image = adjust_sigmoid(image_float)
    filtered_image = gaussian_high_pass(rescaled_image, sigma = 10)
    threshold_abs = threshold_rel * np.max(filtered_image)
    if threshold_auto :
        threshold_abs = threshold_otsu(filtered_image)
    binary_image = filtered_image > threshold_abs
    labeled_image = label(binary_image)
    centroids = regionprops(labeled_image)
    ret = []
    for centroid in centroids :
        ret.append(centroid.centroid)

    retf = np.array(ret)
    return retf,binary_image

def extract_Region_Of_Interest(centroids, crop_factor=5, bead_size=10):
    """ Uses centroids of detected beads to extract region of interest for each and remove the ones overlapped or too near from the edges.

    Parameters
    ----------
    image : np.ndarray
        The image to be processed
    centroids : np.ndarray
        The list of coordinates of each centroid.
    

    Returns
    -------
    Shapes : List of np.ndarray
        List of all shapes representing ROIs
    """
    rois = []
    image_shape = [512,512]
    roi_size = (crop_factor * bead_size) / 2
    for i,centroid in enumerate(centroids) :
        over = False
        for y,c2 in enumerate(centroids) : 
            if i != y :
                if math.dist(centroid,c2)<(math.sqrt(2) * roi_size):
                    over = True
                    break
        if over == False : 
            tmp = np.array([
                    [centroid[1] - roi_size, centroid[2] - roi_size],
                    [centroid[1] - roi_size, centroid[2] + roi_size],
                    [centroid[1] + roi_size, centroid[2] + roi_size],
                    [centroid[1] + roi_size, centroid[2] - roi_size],           
                ]
            )
            overlapped = is_roi_overlapped(rois,tmp)
            if not overlapped :
                if is_roi_in_image(tmp, image_shape):
                    rois.append(tmp)
    return rois

def is_roi_in_image(roi, image_shape):
    height, width = image_shape
    for y,x in roi :
        if y < 0 or y>= height or x < 0 or  x>= width : 
            return False
    return True

def is_roi_overlapped(rois,roi) :
    new_y_min = min(roi[:,0])
    new_y_max = max(roi[:,0])
    new_x_min = min(roi[:,1])
    new_x_max = max(roi[:,1])
    for i,R in enumerate(rois):
        y_min = min(R[:,0])
        y_max = max(R[:,0])
        x_min = min(R[:,1])
        x_max = max(R[:,1])
        no_overlap_x = (new_x_max < x_min) or (x_max < new_x_min)
        no_overlap_y = (new_y_max < y_min) or (y_max < new_y_min)
        if not (no_overlap_x or no_overlap_y):
            return True
    return False
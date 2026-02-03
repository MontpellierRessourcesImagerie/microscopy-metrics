import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log

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


def detect_psf_positions(image, min_distance=5, threshold_rel=0.3):
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
    threshold_abs = threshold_rel * np.max(image)
    coordinates = peak_local_max(image,min_distance=min_distance,threshold_abs=threshold_abs)
    return coordinates

def detect_psf_centroids(image, max_sigma = 3, threshold_rel=0.1):
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
    filtered_image = gaussian_high_pass(image_float, sigma = max_sigma)
    filtered_image[filtered_image < 0] = 0
    
    centroids = []
    if image.ndim >= 2  :
        threshold_abs = threshold_rel * np.max(filtered_image)
        blobs = blob_log(filtered_image, min_sigma=1, max_sigma=max_sigma, threshold=threshold_abs)
        centroids = np.array([[blob[0],blob[1],blob[2]] for blob in blobs])
    else :
        for z in range (image_float.shape[0]):
            slice_2d = filtered_image[z,:,:]
            threshold_abs = threshold_rel * np.max(slice_2d)
            blobs = blob_log(slice_2d,min_sigma=1,
            max_sigma=max_sigma, threshold=threshold_abs)
            for blob in blobs:
                centroids.append([z,blob[0],blob[1]])
        centroids = np.array(centroids)
    return centroids
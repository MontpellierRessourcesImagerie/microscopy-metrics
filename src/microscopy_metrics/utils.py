import numpy as np 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
import math
from concurrent.futures import ThreadPoolExecutor,as_completed


def um_to_px(x,axisPhysicalSize):
    """
    Args:
        x (float): The value in um to convert
        axisPhysicalSize (float):Physical size of a pixel (µm/px)

    Returns:
        float: The number of pixels corresponding to x
    """
    if axisPhysicalSize == 0.0 :
        return 0.0
    x_conv = (x/axisPhysicalSize)
    return x_conv

def px_to_um(x,axisPhysicalSize):
    """
    Args:
        x (float):  The value in pixels to convert
        axisPhysicalSize (float): Physical size of a pixel (µm/px)

    Returns:
        float: The µm value corresponding to x
    """
    x_conv = x * axisPhysicalSize
    return x_conv

def is_roi_not_in_rejection(centroid,image_shape, rejection_zone):
    """
    Args:
        centroid (List): Coordinates of the bead's centroid
        image_shape (List): Dimensions of the picture
        rejection_zone (float): Minimal distance between top/bottom and the centroid
    
    Returns:
        Boolean: True if the bead is not in the rejection zone.
    """
    if ((centroid[0] - rejection_zone) < 0) or ((centroid[0] + rejection_zone) > image_shape[0]) :
        return False
    return True 

def is_roi_in_image(roi, image_shape):
    """
    Args:
        roi (np.array): Coordinates of corners of the ROI
        image_shape (List): Dimensions of the image

    Returns:
        Boolean: True if the ROI is contained in the image shape
    """
    stack,height, width = image_shape
    for z,y,x in roi :
        if y < 0 or y>= height or x < 0 or  x>= width : 
            return False
    return True

def is_roi_overlapped(rois,roi) :
    """
    Args:
        rois (List): List of all rois of the image
        roi (np.array): Coordinates of vertices of the ROI
    
    Returns:
        Boolean: False if the ROI is not overlapped
    """
    new_y_min = min(roi[:,1])
    new_y_max = max(roi[:,1])
    new_x_min = min(roi[:,2])
    new_x_max = max(roi[:,2])
    for i,R in enumerate(rois):
        y_min = min(R[:,1])
        y_max = max(R[:,1])
        x_min = min(R[:,2])
        x_max = max(R[:,2])
        no_overlap_x = (new_x_max < x_min) or (x_max < new_x_min)
        no_overlap_y = (new_y_max < y_min) or (y_max < new_y_min)
        if not (no_overlap_x or no_overlap_y):
            return True
    return False

def legacy_threshold(image,nb_iteration=100):
    """Apply the Metroloj_Qc's 'legacy threshold'

    Args:
        image (np.array): Image to apply the threshold
        nb_iteration (int, optional): Number of iteration to process. Defaults to 100.

    Returns:
        float: Value of the threshold
    """
    img_min = np.min(image)
    img_max = np.max(image)
    midpoint = (img_max - img_min)/2
    image[image < 0] = 0
    for i in range (nb_iteration):
        background = image[image <= midpoint]
        signal = image[image > midpoint]
        mean_background = np.mean(background) if len(background) > 0 else img_min
        mean_signal = np.mean(signal) if len(signal) > 0 else img_max
        n_midpoint = (mean_background + mean_signal)/2
        if abs(n_midpoint - midpoint) < 1e-6 :
            break
        midpoint = n_midpoint
    return midpoint
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from skimage.exposure import adjust_sigmoid
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def umToPx(x, axisPhysicalSize):
    """
    Args:
        x (float): The value in um to convert
        axisPhysicalSize (float):Physical size of a pixel (µm/px)

    Returns:
        float: The number of pixels corresponding to x
    """
    if axisPhysicalSize == 0.0:
        return 0.0
    xConv = x / axisPhysicalSize
    return xConv


def pxToUm(x, axisPhysicalSize):
    """
    Args:
        x (float):  The value in pixels to convert
        axisPhysicalSize (float): Physical size of a pixel (µm/px)

    Returns:
        float: The µm value corresponding to x
    """
    xConv = x * axisPhysicalSize
    return xConv


def isRoiNotInRejection(centroid, imageShape, rejectionZone):
    """
    Args:
        centroid (List): Coordinates of the bead's centroid
        imageShape (List): Dimensions of the picture
        rejectionZone (float): Minimal distance between top/bottom and the centroid

    Returns:
        Boolean: True if the bead is not in the rejection zone.
    """
    if ((centroid[0] - rejectionZone) < 0) or (
            (centroid[0] + rejectionZone) > imageShape[0]
    ):
        return False
    return True


def isRoiInImage(roi, imageShape):
    """
    Args:
        roi (np.array): Coordinates of corners of the ROI
        imageShape (List): Dimensions of the image

    Returns:
        Boolean: True if the ROI is contained in the image shape
    """
    stack, height, width = imageShape
    for z, y, x in roi:
        if y < 0 or y >= height or x < 0 or x >= width:
            return False
    return True


def isRoiOverlapped(rois, roi):
    """
    Args:
        rois (List): List of all rois of the image
        roi (np.array): Coordinates of vertices of the ROI

    Returns:
        Boolean: False if the ROI is not overlapped
    """
    newYMin = min(roi[:, 1])
    newYMax = max(roi[:, 1])
    newXMin = min(roi[:, 2])
    newXMax = max(roi[:, 2])
    for i, R in enumerate(rois):
        yMin = min(R[:, 1])
        yMax = max(R[:, 1])
        xMin = min(R[:, 2])
        xMax = max(R[:, 2])
        noOverlapX = (newXMax < xMin) or (xMax < newXMin)
        noOverlapY = (newYMax < yMin) or (yMax < newYMin)
        if not (noOverlapX or noOverlapY):
            return True
    return False

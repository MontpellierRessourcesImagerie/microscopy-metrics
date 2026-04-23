import os
import math
import numpy as np

from PIL import Image
from scipy.signal import find_peaks
from skimage.draw import polygon_perimeter

from microscopy_metrics.utils import umToPx
from microscopy_metrics.ImageAnalyzer import ImageAnalyzer
from microscopy_metrics.BeadAnalyzer import BeadAnalyzer


class Detection(object):
    """Class for detecting and extracting regions of interest (ROIs) from microscopy images based on detected centroids.
    It initializes the imageAnalyzer object to manage the analysis of the image and the detected beads, and provides methods for setting parameters related to bead detection and ROI extraction.
    """

    def __init__(self, image=None):
        self._image = image
        self._cropFactor = 5
        self._beadSize = 0.6
        self._rejectionDistance = 0.5
        self._pixelSize = [0.1, 0.06, 0.06]
        self._thresholdIntensity = 0.75

        self._imageAnalyzer = None
        self._thresholdTool = None
        self._detectionTool = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("Please, select an Image with 2 or 3 dimensions.")
        self._image = image

    @property
    def cropFactor(self):
        return self._cropFactor

    @cropFactor.setter
    def cropFactor(self, value):
        if not isinstance(value, int) or value > np.max(self._image.shape):
            raise ValueError(
                "Please, choose an integer smaller than image as a crop factor"
            )
        self._cropFactor = value

    @property
    def beadSize(self):
        return self._beadSize

    @beadSize.setter
    def beadSize(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, choose a float as a size of bead")
        self._beadSize = value

    @property
    def rejectionDistance(self):
        return self._rejectionDistance

    @rejectionDistance.setter
    def rejectionDistance(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, choose a float as a rejection distance")
        self._rejectionDistance = value

    @property
    def pixelSize(self):
        return self._pixelSize

    @pixelSize.setter
    def pixelSize(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Shape format not compatible with current image")
        self._pixelSize = value

    @property
    def thresholdTool(self):
        return self._thresholdTool

    @thresholdTool.setter
    def thresholdTool(self, value):
        self._thresholdTool = value

    def getMeanIntensity(self):
        """Calculates the mean intensity of the detected centroids in the image.
        Returns:
            float: The calculated mean intensity of the detected centroids.
        """
        if len(self._imageAnalyzer._beadAnalyzer) > 0:
            result = 0.0
            count = 0
            for bead in self._imageAnalyzer._beadAnalyzer:
                if bead._rejected == False:
                    z, y, x = bead._centroid
                    result += self._image[int(z), int(y), int(x)]
                    count += 1
            return (result / count) if count > 0 else 0.0
        return 0

    def extractRegionOfInterest(self):
        """Extracts regions of interest (ROIs) from the image based on the detected centroids and specified parameters for bead size, crop factor, and rejection distance.
        The method calculates the ROI for each detected bead, checks for overlaps with other beads and the image boundaries, and applies rejection criteria based on intensity and peak detection within the ROI.
        """
        roiSize = umToPx((self._cropFactor * self._beadSize) / 2, self._pixelSize[2])
        for bead in self._imageAnalyzer._beadAnalyzer:
            bead._roi = np.array(
                [
                    [
                        int(bead._centroid[0]),
                        math.ceil(bead._centroid[1] - roiSize),
                        math.ceil(bead._centroid[2] - roiSize),
                    ],
                    [
                        int(bead._centroid[0]),
                        math.ceil(bead._centroid[1] - roiSize),
                        math.ceil(bead._centroid[2] + roiSize),
                    ],
                    [
                        int(bead._centroid[0]),
                        math.ceil(bead._centroid[1] + roiSize),
                        math.ceil(bead._centroid[2] + roiSize),
                    ],
                    [
                        int(bead._centroid[0]),
                        math.ceil(bead._centroid[1] + roiSize),
                        math.ceil(bead._centroid[2] - roiSize),
                    ],
                ]
            )
            for other_bead in self._imageAnalyzer._beadAnalyzer:
                if bead._id != other_bead._id and other_bead._rejected == False:
                    if math.dist(bead._centroid, other_bead._centroid) < (
                        math.sqrt(2) * roiSize
                    ):
                        other_bead._rejected = True
                        other_bead._rejectionDesc = "Overlapped with bead " + str(
                            other_bead._id
                        )
            if not bead._rejected and bead._centroid is not None:
                if self.isRoiOverlapped(bead._id):
                    bead._rejected = True
                    bead._rejectionDesc = "Overlapped with another bead"
                elif not self.isRoiInImage(bead._roi):
                    bead._rejected = True
                    bead._rejectionDesc = "ROI not in image"
                elif not self.isRoiNotInRejection(bead._centroid):
                    bead._rejected = True
                    bead._rejectionDesc = "Centroid in rejection zone"
        meanIntensity = self.getMeanIntensity()
        for bead in self._imageAnalyzer._beadAnalyzer:
            bead._image = self._image[
                ...,
                bead._roi[0][1] : bead._roi[2][1],
                bead._roi[0][2] : bead._roi[1][2],
            ]
            physic = [
                int(bead._centroid[0]),
                int(bead._centroid[1] - bead._roi[0][1]),
                int(bead._centroid[2] - bead._roi[0][2]),
            ]
            if bead._rejected == False:
                z = bead._image[:, physic[1], physic[2]]
                y = bead._image[physic[0], :, physic[2]]
                x = bead._image[physic[0], physic[1], :]
                peaksX, _ = find_peaks(
                    x, height=np.min(x) + (np.max(x) - np.min(x)) * 0.5, distance=3
                )
                peaksY, _ = find_peaks(
                    y, height=np.min(y) + (np.max(y) - np.min(y)) * 0.5, distance=3
                )
                peaksZ, _ = find_peaks(
                    z, height=np.min(z) + (np.max(z) - np.min(z)) * 0.5, distance=3
                )
                if bead._image[int(physic[0]), int(physic[1]), int(physic[2])] < (
                    self._thresholdIntensity * meanIntensity
                ) or bead._image[int(physic[0]), int(physic[1]), int(physic[2])] > (
                    (1 + (1 - self._thresholdIntensity)) * meanIntensity
                ):
                    bead._rejected = True
                    bead._rejectionDesc = "Intensity criteria not met"
                if len(peaksX) > 1 or len(peaksY) > 1 or len(peaksZ) > 1:
                    bead._rejected = True
                    bead._rejectionDesc = "Multiple peaks detected in ROI"

    def isRoiOverlapped(self, index):
        """Checks if the given ROI overlaps with any of the already extracted ROIs.
        Args:
            index (int): The index of the bead for which to check overlap with existing ROIs.
        Returns:
            Boolean: True if the given ROI overlaps with any of the existing ROIs, False otherwise.
        """
        actualBead = self._imageAnalyzer._beadAnalyzer[index]
        roi = actualBead._roi
        newYMin = min(roi[:, 1])
        newYMax = max(roi[:, 1])
        newXMin = min(roi[:, 2])
        newXMax = max(roi[:, 2])
        for bead in self._imageAnalyzer._beadAnalyzer:
            if (
                bead._rejected == False
                and actualBead._id != bead._id
                and bead._roi is not None
            ):
                yMin = min(bead._roi[:, 1])
                yMax = max(bead._roi[:, 1])
                xMin = min(bead._roi[:, 2])
                xMax = max(bead._roi[:, 2])
                noOverlapX = (newXMax < xMin) or (xMax < newXMin)
                noOverlapY = (newYMax < yMin) or (yMax < newYMin)
                if not (noOverlapX or noOverlapY):
                    return True
        return False

    def isRoiNotInRejection(self, centroid):
        """Checks if the given centroid is located within the rejection zone near the top or bottom of the image.
        Args:
            centroid (List): Coordinates of the bead's centroid
            imageShape (List): Dimensions of the picture
            rejectionZone (float): Minimal distance between top/bottom and the centroid
        Returns:
            Boolean: True if the centroid is not located within the rejection zone, False otherwise.
        """
        rejectionZone = math.ceil(umToPx(self._rejectionDistance, self._pixelSize[0]))
        if ((centroid[0] - rejectionZone) < 0) or (
            (centroid[0] + rejectionZone) > self._image.shape[0]
        ):
            return False
        return True

    def isRoiInImage(self, roi):
        """Checks if the given ROI is contained within the boundaries of the image.
        Args:
            roi (np.array): Coordinates of corners of the ROI to be checked for containment within the image boundaries.
        Returns:
            Boolean: True if the given ROI is contained within the image boundaries, False otherwise.
        """
        _, height, width = self._image.shape
        for _, y, x in roi:
            if y < 0 or y >= height or x < 0 or x >= width:
                return False
        return True

    def run(self, outputDir=None, cropPsf=True):
        """Runs the complete detection workflow, including detecting centroids, extracting ROIs, and saving cropped PSF images.
        Args:
            outputDir (Path, optional): Directory of the output folder where cropped PSF images will be saved. Required if cropPsf is set to True. Defaults to None.
            cropPsf (bool, optional): Flag indicating whether to crop PSF images and save them in the output directory. Defaults to True.
        Raises:
            ValueError: If cropPsf is set to True and outputDir is not provided, indicating that the output directory is required for saving cropped PSF images.
        Yields:
            String: Description of the current step in the detection workflow, providing progress updates to the user.
        """
        if outputDir is None and cropPsf == True:
            raise ValueError(
                "Output directory is required for saving cropped PSF images."
            )
        self._detectionTool._image = self.image
        self._imageAnalyzer = ImageAnalyzer(
            image=self._image,
            beadSize=self._beadSize,
            pixelSize=self._pixelSize,
            BeadAnalyzer=[],
        )
        self._detectionTool.detect()
        for i, centroid in enumerate(self._detectionTool._centroids):
            bead = BeadAnalyzer(id=i, centroid=centroid)
            self._imageAnalyzer._beadAnalyzer.append(bead)
        yield {"desc": "Extracting Rois..."}
        self.extractRegionOfInterest()
        if cropPsf:
            yield {"desc": "Cropping PSFs..."}
            self.cropPsf(outputDir)

    def getActivePath(self, index, outputDir):
        """Provides the path to the folder corresponding to the selected bead, creating it if it does not exist.
        Args:
            index (int): The index of the bead for which to get the active path
            outputDir (Path): The directory of the output folder where the bead's folder will be created if it does not exist

        Returns:
            Path: The path to the folder corresponding to the selected bead
        """
        activePath = os.path.join(outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def addRoiOnImage(self, roi):
        """Adds a visual representation of the ROI on the image by drawing a polygon perimeter around the specified ROI coordinates.
        Args:
            roi (np.ndarray): Coordinates of the corners of the ROI to be highlighted on the image.

        Returns:
            np.ndarray: The image with the ROI highlighted by a polygon perimeter.
        """
        if self._image.ndim == 3:
            imageTmp = np.max(self._image, axis=0)
        imageTmp = (
            (imageTmp - imageTmp.min()) / (imageTmp.max() - imageTmp.min()) * 255
        ).astype(np.uint8)
        imageRGB = np.stack([imageTmp, imageTmp, imageTmp], axis=-1)
        rr, cc = polygon_perimeter(
            [roi[0, 1], roi[1, 1], roi[2, 1], roi[3, 1]],
            [roi[0, 2], roi[1, 2], roi[2, 2], roi[3, 2]],
            imageTmp.shape,
        )
        imageRGB[rr, cc, 0] = 255
        imageRGB[rr, cc, 1] = 255
        imageRGB[rr, cc, 2] = 255
        return imageRGB

    def cropPsf(self, outputDir):
        """Crops the PSF images for each valid ROI and saves them in the specified output directory.
        Args:
            outputDir (Path): The directory of the output folder where the cropped PSF images will be saved.
        """
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                roi = bead._roi
                data = bead._image
                activePath = self.getActivePath(bead._id, outputDir)
                physic = [
                    int(bead._centroid[0]),
                    int(bead._centroid[1] - bead._roi[0][1]),
                    int(bead._centroid[2] - bead._roi[0][2]),
                ]
                imageFloat = data.astype(np.float32)
                imageFloat = (imageFloat - np.min(imageFloat)) / (
                    np.max(imageFloat) - np.min(imageFloat) + 1e-6
                )
                imageFloat[imageFloat < 0] = 0
                imageUint16 = (imageFloat * 255).astype(np.uint8)
                XYData = Image.fromarray(imageUint16[physic[0], :, :])
                YZData = Image.fromarray(imageUint16[:, :, physic[2]])
                XZData = Image.fromarray(imageUint16[:, physic[1], :])
                XYData.save(os.path.join(activePath, "XY_view.png"))
                YZData.save(os.path.join(activePath, "YZ_view.png"))
                XZData.save(os.path.join(activePath, "XZ_view.png"))
                imageRoi = Image.fromarray(self.addRoiOnImage(roi))
                imageRoi.save(os.path.join(activePath, "Localisation.png"))

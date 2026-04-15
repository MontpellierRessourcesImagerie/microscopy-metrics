import os
import math
import numpy as np

from PIL import Image
from scipy.signal import find_peaks
from skimage.draw import polygon_perimeter

from microscopy_metrics.utils import umToPx
from microscopy_metrics.detectionTools.detection_tool import DetectionTool
from microscopy_metrics.thresholdTools.threshold_tool import Threshold


class Detection(object):
    """Class for detecting and extracting regions of interest (ROIs) from microscopy images based on detected centroids.
    This class provides methods for detecting centroids using a specified detection tool, extracting ROIs around the detected centroids, and saving the cropped PSF images for further analysis.
    It includes properties for configuring the detection parameters, such as crop factor, sigma, minimum distance, bead size, rejection distance, pixel size, and threshold intensity.
    The class also includes methods for checking ROI overlap, validating ROI positions within the image, and running the complete detection workflow.
    """

    def __init__(self, image=None):
        self._image = image
        self._cropFactor = 5
        self._beadSize = 0.6
        self._rejectionDistance = 0.5
        self._pixelSize = [0.1, 0.06, 0.06]
        self._thresholdIntensity = 0.75

        self._thresholdTool = None
        self._detectionTool = None

        self._normalizedImage = None
        self._highPassedImage = None
        self._centroids = []
        self._roisExtracted = []
        self._listIdCentroidsRetained = []
        self._cropped = []

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
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if not isinstance(value, int):
            raise ValueError("Please, choose an integer as a sigma")
        self._sigma = value

    @property
    def minDistance(self):
        return self._minDistance

    @minDistance.setter
    def minDistance(self, value):
        if not isinstance(value, int):
            raise ValueError("Please, choose an integer as a minimal distance")
        self._minDistance = value

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

    def getMeanIntensity(self, centroids):
        """Calculates the mean intensity of the detected centroids in the image.
        Args:
            centroids (List): List of detected centroids for which to calculate the mean intensity.
        Returns:
            float: The calculated mean intensity of the detected centroids.
        """
        if len(centroids) > 0:
            result = 0.0
            for i in centroids:
                z, y, x = i
                result += self._image[int(z), int(y), int(x)]
            return result / len(centroids)
        return 0

    def extractRegionOfInterest(self):
        """Extracts regions of interest (ROIs) around the detected centroids in the image.
        The method retrieves the detected centroids from the detection tool, calculates the ROI coordinates based on the specified crop factor and bead size, and checks for overlapping ROIs and their positions within the image.
        It also applies intensity-based filtering to retain only valid ROIs based on the mean intensity of the detected centroids and the specified threshold intensity. The valid ROIs are stored in the class attributes for further processing and analysis.
        The method ensures that the extracted ROIs are non-overlapping, contained within the image boundaries, and not located within the rejection zone near the top or bottom of the image.
        The final list of extracted ROIs and corresponding centroids are retained for subsequent cropping and analysis steps in the detection workflow.
        The method also includes checks to ensure that the extracted ROIs contain only a single bead based on the intensity profiles along the three axes, and removes any ROIs that contain multiple peaks or do not meet the intensity criteria.
        """
        roiSize = umToPx((self._cropFactor * self._beadSize) / 2, self._pixelSize[2])
        for i, centroid in enumerate(self._centroids):
            over = False
            for y, c2 in enumerate(self._centroids):
                if i != y:
                    if math.dist(centroid, c2) < (math.sqrt(2) * roiSize):
                        over = True
                        break
            if over == False:
                tmp = np.array(
                    [
                        [
                            int(centroid[0]),
                            math.ceil(centroid[1] - roiSize),
                            math.ceil(centroid[2] - roiSize),
                        ],
                        [
                            int(centroid[0]),
                            math.ceil(centroid[1] - roiSize),
                            math.ceil(centroid[2] + roiSize),
                        ],
                        [
                            int(centroid[0]),
                            math.ceil(centroid[1] + roiSize),
                            math.ceil(centroid[2] + roiSize),
                        ],
                        [
                            int(centroid[0]),
                            math.ceil(centroid[1] + roiSize),
                            math.ceil(centroid[2] - roiSize),
                        ],
                    ]
                )
                overlapped = self.isRoiOverlapped(tmp)
                if not overlapped:
                    if self.isRoiInImage(tmp) and self.isRoiNotInRejection(centroid):
                        self._roisExtracted.append(tmp)
                        self._listIdCentroidsRetained.append(i)
        retainedCentroids = [self._centroids[i] for i in self._listIdCentroidsRetained]
        meanIntensity = self.getMeanIntensity(retainedCentroids)
        tmpRoisExtracted = []
        tmpListIDCentroidsRetained = []
        for i, centroid in enumerate(retainedCentroids):
            roi_image = self._image[
                ...,
                self._roisExtracted[i][0][1] : self._roisExtracted[i][2][1],
                self._roisExtracted[i][0][2] : self._roisExtracted[i][1][2],
            ]
            centroidIdx = self._listIdCentroidsRetained[i]
            physic = [
                int(self._centroids[centroidIdx][0]),
                int(self._centroids[centroidIdx][1] - self._roisExtracted[i][0][1]),
                int(self._centroids[centroidIdx][2] - self._roisExtracted[i][0][2]),
            ]
            z = roi_image[:, physic[1], physic[2]]
            y = roi_image[physic[0], :, physic[2]]
            x = roi_image[physic[0], physic[1], :]
            height = Threshold.getInstance("legacy").getThreshold(roi_image)
            peaksX, _ = find_peaks(
                x, height=np.min(x) + (np.max(x) - np.min(x)) * 0.5, distance=3
            )
            peaksY, _ = find_peaks(
                y, height=np.min(y) + (np.max(y) - np.min(y)) * 0.5, distance=3
            )
            peaksZ, _ = find_peaks(
                z, height=np.min(z) + (np.max(z) - np.min(z)) * 0.5, distance=3
            )
            if not (
                self._image[int(centroid[0]), int(centroid[1]), int(centroid[2])]
                < (self._thresholdIntensity * meanIntensity)
                or self._image[int(centroid[0]), int(centroid[1]), int(centroid[2])]
                > ((1 + (1 - self._thresholdIntensity)) * meanIntensity)
            ):
                if not (len(peaksX) > 1 or len(peaksY) > 1 or len(peaksZ) > 1):
                    tmpRoisExtracted.append(self._roisExtracted[i])
                    tmpListIDCentroidsRetained.append(self._listIdCentroidsRetained[i])
                else:
                    print(f"ROI {i} removed because of two beads detected")
            else:
                print(f"ROI {i} removed because of intensity")
        if len(tmpListIDCentroidsRetained) > 0:
            self._roisExtracted = tmpRoisExtracted
            self._listIdCentroidsRetained = tmpListIDCentroidsRetained

    def isRoiOverlapped(self, roi):
        """Checks if the given ROI overlaps with any of the already extracted ROIs.
        The method compares the coordinates of the given ROI with the coordinates of the already extracted ROIs to determine if there is any overlap.
        It checks for both horizontal and vertical overlaps by comparing the minimum and maximum coordinates of the ROIs.
        Args:
            roi (np.array): Coordinates of the corners of the ROI to be checked for overlap with existing ROIs.
        Returns:
            Boolean: True if the given ROI overlaps with any of the existing ROIs, False otherwise.
        """
        newYMin = min(roi[:, 1])
        newYMax = max(roi[:, 1])
        newXMin = min(roi[:, 2])
        newXMax = max(roi[:, 2])
        for i, R in enumerate(self._roisExtracted):
            yMin = min(R[:, 1])
            yMax = max(R[:, 1])
            xMin = min(R[:, 2])
            xMax = max(R[:, 2])
            noOverlapX = (newXMax < xMin) or (xMax < newXMin)
            noOverlapY = (newYMax < yMin) or (yMax < newYMin)
            if not (noOverlapX or noOverlapY):
                return True
        return False

    def isRoiNotInRejection(self, centroid):
        """Checks if the given centroid is located within the rejection zone near the top or bottom of the image.
        The method calculates the rejection zone based on the specified rejection distance and pixel size, and checks if the centroid's z-coordinate is within the rejection zone.
        It ensures that the detected centroids are not located too close to the edges of the image, which could lead to inaccurate ROI extraction and analysis.
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
        The method compares the coordinates of the given ROI with the dimensions of the image to determine if the ROI is fully contained within the image boundaries.
        This ensures that the extracted ROIs are valid and can be properly analyzed without encountering issues related to out-of-bounds errors or incomplete data.
        Args:
            roi (np.array): Coordinates of corners of the ROI to be checked for containment within the image boundaries.
        Returns:
            Boolean: True if the given ROI is contained within the image boundaries, False otherwise.
        """
        stack, height, width = self._image.shape
        for z, y, x in roi:
            if y < 0 or y >= height or x < 0 or x >= width:
                return False
        return True

    def run(self, outputDir=None, cropPsf=True):
        """Runs the complete detection workflow, including detecting centroids, extracting ROIs, and saving cropped PSF images.
        The method orchestrates the entire detection process by first invoking the detection tool to identify centroids in the image, then extracting ROIs around the detected centroids while ensuring non-overlapping and valid ROIs, and finally saving the cropped PSF images for each valid ROI in the specified output directory.
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
        self._centroids = []
        self._roisExtracted = []
        self._listIdCentroidsRetained = []
        self._cropped = []
        self._detectionTool._image = self.image
        self._detectionTool.detect()
        self._centroids = self._detectionTool._centroids
        yield {"desc": "Extracting Rois..."}
        self.extractRegionOfInterest()
        if cropPsf:
            yield {"desc": "Cropping PSFs..."}
            self.cropPsf(outputDir)

    def getActivePath(self, index, outputDir):
        """Provides the path to the folder corresponding to the selected bead, creating it if it does not exist.
        The method constructs the path to the folder for the selected bead based on the provided index and output directory.
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
        The method iterates through the list of extracted ROIs and corresponding centroids, crops the PSF images based on the ROI coordinates, and saves the cropped images in the output directory with appropriate naming conventions.
        It also adds a visual representation of the ROI on the cropped images for better visualization and understanding of the extracted regions.
        Args:
            outputDir (Path): The directory of the output folder where the cropped PSF images will be saved.
        """
        for i, roi in enumerate(self._roisExtracted):
            data = self._image[..., roi[0][1] : roi[2][1], roi[0][2] : roi[1][2]]
            self._cropped.append(data)
            activePath = self.getActivePath(i, outputDir)
            centroidIdx = self._listIdCentroidsRetained[i]
            physic = [
                int(self._centroids[centroidIdx][0]),
                int(self._centroids[centroidIdx][1] - self._roisExtracted[i][0][1]),
                int(self._centroids[centroidIdx][2] - self._roisExtracted[i][0][2]),
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

import numpy as np
from .utils import *
import math
import os
from PIL import Image
from skimage.draw import polygon_perimeter
from .detection_tool import DetectionTool,PeakLocalMaxDetector
from .threshold_tool import Threshold
from .fitting import Fitting1D
from scipy.signal import find_peaks

class Detection(object):
    """Standard class for operations relative to detection and extraction of PSFs"""

    def __init__(self, image=None):
        self._image = image
        self._cropFactor = 5
        self._sigma = 3
        self._minDistance = 1
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
        if len(centroids) > 0 :
            result = 0.0
            for i in centroids :
                z,y,x = i
                result += self._image[int(z),int(y),int(x)]
            return result/len(centroids) 
        return 0

    def extractRegionOfInterest(self):
        """Uses found _centroids to extract region of interest
        Automatically rejects the ones overlapped or too near from the edges.
        """
        roiSize = umToPx(
            (self._cropFactor * self._beadSize) / 2, self._pixelSize[2]
        )
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
                overlapped = isRoiOverlapped(self._roisExtracted, tmp)
                if not overlapped:
                    if isRoiInImage(tmp, self._image.shape) and isRoiNotInRejection(centroid,self._image.shape,math.ceil(umToPx(self._rejectionDistance, self._pixelSize[0]))):
                        self._roisExtracted.append(tmp)
                        self._listIdCentroidsRetained.append(i)
        retainedCentroids = [self._centroids[i] for i in self._listIdCentroidsRetained]
        meanIntensity = self.getMeanIntensity(retainedCentroids)
        tmpRoisExtracted = []
        tmpListIDCentroidsRetained = []
        for i,centroid in enumerate(retainedCentroids) :
            roi_image = self._image[..., self._roisExtracted[i][0][1] : self._roisExtracted[i][2][1], self._roisExtracted[i][0][2] : self._roisExtracted[i][1][2]]
            centroidIdx = self._listIdCentroidsRetained[i]
            physic = [
                int(self._centroids[centroidIdx][0]),
                int(self._centroids[centroidIdx][1] - self._roisExtracted[i][0][1]),
                int(self._centroids[centroidIdx][2] - self._roisExtracted[i][0][2]),
            ]
            z = roi_image[:,physic[1],physic[2]]
            y = roi_image[physic[0],:,physic[2]]
            x = roi_image[physic[0],physic[1],:]
            height = Threshold.getInstance("legacy").getThreshold(roi_image)
            peaksX,_ = find_peaks(x,height=np.min(x)+(np.max(x)-np.min(x))*0.5, distance=3)
            peaksY,_ = find_peaks(y,height=np.min(y)+(np.max(y)-np.min(y))*0.5, distance=3)
            peaksZ,_ = find_peaks(z,height=np.min(z)+(np.max(z)-np.min(z))*0.5, distance=3)
            if not(self._image[int(centroid[0]),int(centroid[1]),int(centroid[2])] < (self._thresholdIntensity * meanIntensity) or self._image[int(centroid[0]),int(centroid[1]),int(centroid[2])] > ((1+(1-self._thresholdIntensity)) * meanIntensity)) :
                if not(len(peaksX) > 1 or len(peaksY) >1 or len(peaksZ) > 1) :
                    tmpRoisExtracted.append(self._roisExtracted[i])
                    tmpListIDCentroidsRetained.append(self._listIdCentroidsRetained[i])
                else:
                    print(f"ROI {i} removed because of two beads detected")
            else :
                print(f"ROI {i} removed because of intensity")
        if len(tmpListIDCentroidsRetained) > 0 :
            self._roisExtracted = tmpRoisExtracted
            self._listIdCentroidsRetained = tmpListIDCentroidsRetained
            

    def run(self, outputDir=None, cropPsf=True):
        """Function to operate complete detection workflow

        Args:
            outputDir (Path, optional): Directory of the output folder. Defaults to None.
            cropPsf (bool, optional): Allow or not the generation of _cropped PSF images. Defaults to True.

        Raises:
            ValueError: To generate images of the _cropped PSFs, the outputDir have to exist

        Yields:
            String: Return the current step of the workflow
        """
        if outputDir is None and cropPsf == True:
            raise ValueError("Problem to find output folder")
        self._centroids = []
        self._roisExtracted = []
        self._listIdCentroidsRetained = []
        self._cropped = []
        self._detectionTool.detect()
        self._centroids = self._detectionTool._centroids
        yield {"desc": "Extracting Rois..."}
        self.extractRegionOfInterest()
        if cropPsf:
            yield {"desc": "Cropping PSFs..."}
            self.cropPsf(outputDir)

    def getActivePath(self, index, outputDir):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list
            outputDir (Path): Directory of the output folder

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        activePath = os.path.join(outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def addRoiOnImage(self, roi):
        """Function to draw a square representating an ROI in a picture

        Args:
            roi (np.array): List of the four corners coordinates of the ROI

        Returns:
            np.ndarray: Modified image with the ROI
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
        """Function to crop image for each ROI and save them

        Args:
            outputDir (Path): Directory of the output folder
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

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from sklearn.metrics import r2_score
from skimage.measure import regionprops, label
from skimage.filters import gaussian
from skimage.segmentation import chan_vese
from skimage.measure import find_contours

from microscopy_metrics.utils import umToPx,pxToUm
from microscopy_metrics.thresholdTools.legacy import ThresholdLegacy
from microscopy_metrics.fittingTools.fitting2D import Fitting2D


class MetricTool(object):
    """Class for calculating various metrics related to microscopy images, such as signal-to-background ratio (SBR), lateral asymmetry ratio (LAR), and sphericity ratio."""

    def __init__(self):
        self._image = None
        self._ringInnerDistance = 1.0
        self._ringThickness = 2.0
        self._pixelSize = [1, 1, 1]

        self._SBR = 0
        self._LAR = 0
        self._sphericity = 0
        self._volume = 0
        self._surface = 0
        self._comaticity = 0
        self._sphericalAberration = 0

    def setNormalizedImage(self, image: np.ndarray) -> np.ndarray:
        """Normalizes the input image to a range of [0, 1] and ensures that all values are non-negative.
        Args:
            image (np.ndarray): The input image to be normalized, which should be a 2D or 3D array representing the microscopy image data.
        Raises:
            ValueError: If the input image is not 2D or 3D.
        Returns:
            np.ndarray: The normalized image with values in the range [0, 1].
        """
        if image.ndim not in (2, 3):
            raise ValueError("Image have to be in 2D or 3D.")
        imageFloat = image.astype(np.float64)
        imageFloat = (imageFloat - np.min(imageFloat)) / (
            np.max(imageFloat) - np.min(imageFloat) + 1e-6
        )
        imageFloat[imageFloat < 0] = 0
        return imageFloat

    def processSingleSBRRing(self):
        """Calculates the signal-to-background ratio (SBR) for a single microscopy image using a ring-based method.
        The calculated SBR values are stored in the class attributes for further analysis and evaluation.
        Raises:
            ValueError: If the input image is not 2D or 3D, if there are no background pixels detected, or if there are no signal pixels detected in the image.
        """
        if self._image.ndim not in (2, 3):
            print("Incorrect picture format")
            return -1
        if self._image.size == 0:
            print("Image is empty")
            return -1
        imageFloat = self.setNormalizedImage(self._image)
        imageFloat = median_filter(imageFloat, size=5)
        thresholdAbs = ThresholdLegacy(nb_iteration=1000).getThreshold(imageFloat)
        binaryImage = imageFloat > thresholdAbs
        labeledImage = label(binaryImage)
        regions = regionprops(labeledImage)
        if not regions:
            return -1
        largestRegion = max(regions, key=lambda r: r.area)
        minZ, minY, minX, maxZ, maxY, maxX = largestRegion.bbox
        center = largestRegion.centroid
        diameterZ = maxZ - minZ
        diameterY = maxY - minY
        diameterX = maxX - minX
        diameterBead = max(diameterZ, diameterY, diameterX)
        innerDistance = (
            umToPx(self._ringInnerDistance, self._pixelSize[2]) + diameterBead / 2
        )
        outerDistance = umToPx(self._ringThickness, self._pixelSize[2]) + innerDistance
        signal = 0.0
        nSignal = 0
        background = 0.0
        nBackground = 0
        for z in range(binaryImage.shape[0]):
            for y in range(binaryImage.shape[1]):
                for x in range(binaryImage.shape[2]):
                    distance = np.sqrt(
                        (z - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (x - center[2]) ** 2
                    )
                    if binaryImage[z, y, x] == 1:
                        nSignal += 1
                        signal += self._image[z, y, x]
                    else:
                        if innerDistance <= distance <= outerDistance:
                            nBackground += 1
                            background += self._image[z, y, x]
        if nBackground == 0:
            raise ValueError("There are no background pixel detected")
        meanBackground = background / nBackground
        if nSignal == 0:
            raise ValueError("There are no signal pixel detected")
        meanSignal = signal / nSignal
        self._SBR = float(meanSignal / meanBackground)

    def lateralAsymmetryRatio(self, FWHM: list):
        """Calculates the lateral asymmetry ratio (LAR) for a PSF based on the full width at half maximum (FWHM) values.
        Args:
            FWHM (list): A list of FWHM values for the PSF.
        Raises:
           ValueError: If the FWHM values are not available or if there are not enough FWHM values to calculate the LAR.
        """
        if FWHM == []:
            raise ValueError(
                "FWHM values are not available or insufficient to calculate LAR."
            )
        tmp = np.array([FWHM[1], FWHM[2]])
        self._LAR = tmp.min() / tmp.max()

    def getVolume(self):
        """Calculates the volume of the object in the microscopy image based on the voxel count and voxel volume.

        Returns:
            float: The calculated volume of the object.
        """
        voxelVolume = self._pixelSize[0] * self._pixelSize[1] * self._pixelSize[2]
        imageFloat = self.setNormalizedImage(self._image)
        imageFloat = median_filter(imageFloat, size=5)
        thresholdAbs = ThresholdLegacy(nb_iteration=1000).getThreshold(imageFloat)
        binaryImage = imageFloat > thresholdAbs
        voxelCount = np.sum(binaryImage)
        volume = voxelCount * voxelVolume
        return volume

    def updateTile(self, tile, tileIndex):
        """Updates the tile configuration based on the provided tile index, which represents the presence or absence of voxels in a 2x2x2 neighborhood.

        Args:
            tile (list): A 3D list representing the tile.
            tileIndex (int): The index of the tile.
        """
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    tile[z][y][x] = (tileIndex & (1 << (z * 4 + y * 2 + x))) > 0

    def surfaceAreaLutD3(self):
        """Calculates the surface area lookup table for 3D tiles.

        Returns:
            list: A list of surface area values for each tile configuration.
        """
        d1 = self._pixelSize[0]
        d2 = self._pixelSize[1]
        d3 = self._pixelSize[2]
        vol = d1 * d2 * d3
        nbConfigs = 256
        tab = [0.0 for _ in range (nbConfigs)]
        im = [
            [[0,0],[0,0]],
            [[0,0],[0,0]]
        ]
        for iConfig in range(nbConfigs):
            self.updateTile(im,iConfig)
            for z in range(2):
                for y in range(2):
                    for x in range(2):
                        if not im[z][y][x]:
                            continue
                        ke1 = 0 if im[z][y][1-x] else vol / d1 / 2.0
                        ke2 = 0 if im[z][1-y][x] else vol / d2 / 2.0
                        ke3 = 0 if im[1-z][y][x] else vol / d3 / 2.0
                        tab[iConfig] += (ke1 + ke2 + ke3) / 3.0
        return tab

    def configIndex(self, configValues):
        """Calculates the index of a tile configuration.
        Args:
            configValues (list): A list of boolean values representing the presence or absence of voxels in a 2x2x2 neighborhood.
        Returns:
            int: The index of the tile configuration.
        """
        return sum(val << i for i, val in enumerate(configValues))

    def getHistogram3D(self):
        """Calculates the histogram of 3D tile configurations in the binary image.
        Returns:
            list: A list of counts for each tile configuration.
        """
        histo = [0 for _ in range(256)]
        imageFloat = self.setNormalizedImage(self._image)
        imageFloat = median_filter(imageFloat, size=5)
        thresholdAbs = ThresholdLegacy(nb_iteration=1000).getThreshold(imageFloat)
        binaryImage = imageFloat > thresholdAbs

        sizeZ, sizeY, sizeX = binaryImage.shape
        configValues = [False for _ in range(8)]
        for z in range(sizeZ+1):
            for y in range(sizeY+1):
                configValues[0] = False
                configValues[2] = False
                configValues[4] = False
                configValues[6] = False
                for x in range(sizeX+1):
                    if x < sizeX:
                        configValues[1] = bool(binaryImage[z - 1, y - 1, x]) if (y > 0 and z > 0) else False
                        configValues[3] = bool(binaryImage[z - 1, y, x]) if (y < sizeY and z > 0) else False
                        configValues[5] = bool(binaryImage[z, y - 1, x]) if (y > 0 and z < sizeZ) else False
                        configValues[7] = bool(binaryImage[z, y, x]) if (y < sizeY and z < sizeZ) else False
                    else : 
                        configValues[1] = configValues[3] = configValues[5] = configValues[7] = False
                    index = self.configIndex(configValues)
                    histo[index] += 1
                    configValues[0] = configValues[1]
                    configValues[2] = configValues[3]
                    configValues[4] = configValues[5]
                    configValues[6] = configValues[7]
        return histo

            
    def applyLut(self, histogram, lut):
        """Applies the surface area lookup table to the histogram of tile configurations to calculate the total surface area.
        Args:
            histogram (list): A list of counts for each tile configuration.
            lut (list): A list of surface area values for each tile configuration.
        Returns:
            float: The total surface area calculated by applying the lookup table to the histogram.
        """
        sum = 0
        for i in range(len(histogram)):
            sum += histogram[i] * lut[i]
        return sum

    def getSurface(self):
        """Calculates the surface area of the object in the microscopy image.
        Returns:
            float: The calculated surface area of the object.
        """
        lut = self.surfaceAreaLutD3()
        histo = self.getHistogram3D()
        return self.applyLut(histo, lut)

    def sphericity(self):
        """Calculates the sphericity of the object in the microscopy image based on its volume and surface area, using the formula: sphericity = (36 * π * volume^2) / (surface^3).
        Returns:
            float: The calculated sphericity of the object.
        """
        c = 36 * math.pi
        self._volume = self.getVolume()
        self._surface = self.getSurface()
        if self._surface == 0:
            self._sphericity = 0.0
            return 0.0            
        self._sphericity = (c * (self._volume**2)) / (self._surface**3)
        return self._sphericity

    def distance(self, x1, x2):
        """Calculates the distance between two points, x1 and x2.
        Args:
            x1 (float): The first point.
            x2 (float): The second point.
        Returns:            
            float: The calculated distance between the two points.
        """
        return abs(x1 - x2)

    def getContours(self, image):
        """Calculates the contours of the object in the microscopy image using the Chan-Vese active contour model.
        The method applies a Gaussian filter to smooth the input image, then uses the Chan-Vese algorithm to segment the image and find the contours of the detected object. 
        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The detected contours.
        """
        smoothImage = gaussian(image, sigma=1.0)
        cvMask = chan_vese(
            smoothImage, 
            mu=0.1, 
            lambda1=1.15, 
            lambda2=1, 
            tol=1e-3, 
            max_num_iter=200, 
            dt=0.5, 
            init_level_set="checkerboard"
        )
        contours = find_contours(cvMask, 0.5)
        if not contours:
            return np.array([])
        snake = max(contours, key=len)
        return snake
    
    def computeComaticityCentroids(self):
        """Calculates the comaticity centroids for the object in the microscopy image by analyzing each image along the Z-axis and determining the maximum drift in the X and Y directions from the center of the image."""
        threshold = ThresholdLegacy(nb_iteration=1000).getThreshold(self._image)
        centroids = []
        maxDriftX = 0.0
        maxDriftY = 0.0 
        for z in range(self._image.shape[0]):
            image = self._image[z]
            imageBinary = image > threshold
            if np.sum(imageBinary) == 0:
                continue
            props = regionprops(label(imageBinary))
            if not props:
                continue
            largestRegion = max(props, key=lambda r: r.area)
            centroids.append(largestRegion.centroid)
            if len(centroids) > 1:
                driftX = pxToUm(abs(centroids[-1][1] - image.shape[1]//2), self._pixelSize[2])
                driftY = pxToUm(abs(centroids[-1][0] - image.shape[0]//2), self._pixelSize[1])
                if driftX > maxDriftX:
                    maxDriftX = driftX
                if driftY > maxDriftY:
                    maxDriftY = driftY
        self._comaticityCentroids = (maxDriftX, maxDriftY)

    def _computeAxisComaticity(self, image, pixelSize):
        """Calculates the axis comaticity for a given 1D image by analyzing the intensity profiles along the specified axis and comparing them to the detected contours of the object in the image.
        Args:
            image (numpy.ndarray): The input image.
            pixelSize (float): The size of each pixel in the image.
        Returns:
            float: The calculated axis comaticity.
        """
        snake = self.getContours(image)
        if snake.size == 0:
            return 0.0
        ndi.gaussian_filter(image, sigma=1.0, output=image)
        score = 0.0
        total = 0
        margin = 3
        for i in range(image.shape[1]):
            if i < snake[:,1].min() - margin or i > snake[:,1].max() + margin:
                continue
            profile = image[:,i]
            amp = float(np.max(profile) - np.min(profile))
            prominenceMin = amp * float(0.4)
            peaks, props = find_peaks(profile,prominence=prominenceMin, distance=2)
            if len(peaks) > 1 and (any(peaks > image.shape[0]/2) and any(peaks < image.shape[0]/2)):
                if all(peaks > image.shape[0]/2):
                    score += pxToUm(self.distance(i, image.shape[1]/2), pixelSize) / (pxToUm(self.distance(peaks[0], peaks[-1]), self._pixelSize[0]))
                else : 
                    score -= pxToUm(self.distance(i, image.shape[1]/2), pixelSize) /(pxToUm(self.distance(peaks[0], peaks[-1]), self._pixelSize[0]))
                total += 1
        if total > 0:
            score /= total
        return score

    def comaticity(self):
        """Calculates the comaticity of the object in the microscopy image by analyzing the intensity profiles along the X and Y axes and comparing them to the detected contours of the object in the image. 
        The method computes the comaticity score for both axes and combines them to provide an overall comaticity score for the object."""
        self.computeComaticityCentroids()
        if self._comaticityCentroids[0] >= self._pixelSize[2] or self._comaticityCentroids[1] >= self._pixelSize[1]:
            image = self._image[:,int(self._image.shape[1]/2),:]
            XScore = self._computeAxisComaticity(image, self._pixelSize[2])
            image = self._image[:,:,int(self._image.shape[2]/2)]
            YScore = self._computeAxisComaticity(image, self._pixelSize[1])
            self._comaticity = np.sqrt(XScore**2 + YScore**2)
        else : 
            self._comaticity = 0.0

    def sphericalAberration(self):
        """Calculates the spherical aberration of the object in the microscopy image by comparing the two halves of the intensity profile along the Z-axis to assess the symmetry of the point spread function (PSF) and determine the degree of spherical aberration present in the image."""
        profile = self._image[:,int(self._image.shape[1]/2),int(self._image.shape[2]/2)]
        ndi.gaussian_filter(profile, sigma=1.0, output=profile)
        zCenter = np.argmax(profile)
        length = min(zCenter, len(profile) - 1 - zCenter)
        before = profile[zCenter-length:zCenter][::-1]
        after = profile[zCenter+1:zCenter+1+length]
        self._sphericalAberration = 1-r2_score(before, after)

    def getFWHM(self, image,mu,sigma):
        """Calculates the full width at half maximum (FWHM) of the point spread function (PSF) in the given image by fitting a 2D Gaussian function to the intensity profile and extracting the FWHM values along the Y and X axes based on the fitted parameters.
        Args:
            image (numpy.ndarray): The input image containing the PSF to be analyzed.
            mu (list): A list of mean values for the Gaussian fit along the Z, Y, and X axes.
            sigma (list): A list of standard deviation values for the Gaussian fit along the Z, Y, and X axes.
        Returns:
            list: A list containing the FWHM values along the Y and X axes.
        """
        amp = np.max(image) - np.min(image)
        bg = np.min(image)
        mu2 = [mu[1], mu[2]]
        sigma2 = [sigma[1], sigma[2]]
        fittingTool = Fitting2D()
        fittingTool._spacing = self._pixelSize
        params, _ = fittingTool.fitCurve(amp,bg,mu2,sigma2,fittingTool.getCoords(image),image)
        fwhmY = fittingTool.fwhm(params[4])
        fwhmX = fittingTool.fwhm(params[5])
        return [fwhmY, fwhmX]

    def astigmatism(self,mu,sigma):
        """Calculates the astigmatism of the object in the microscopy image by comparing the FWHM values along the Y and X axes for two points on either side of the object.
        Args:
            mu (list): A list of mean values for the Gaussian fit along the Z, Y, and X axes.
            sigma (list): A list of standard deviation values for the Gaussian fit along the Z, Y, and X axes.
        """
        fwhmZ = 2 * np.sqrt(2 * np.log(2)) * sigma[0]
        pointBefore = [mu[0] - fwhmZ/2,mu[1], mu[2]]
        pointAfter = [mu[0] + fwhmZ/2,mu[1], mu[2]]
        imageBefore = self._image[int(pointBefore[0])]
        imageAfter = self._image[int(pointAfter[0])]
        fwhmsBefore = self.getFWHM(imageBefore,mu,sigma)
        fwhmsAfter = self.getFWHM(imageAfter,mu,sigma)
        scoreBefore  =  (fwhmsBefore[0] - fwhmsBefore[1]) / (fwhmsBefore[0] + fwhmsBefore[1])
        scoreAfter = (fwhmsAfter[0] - fwhmsAfter[1]) / (fwhmsAfter[0] + fwhmsAfter[1])
        self._astigmatism = abs(scoreAfter - scoreBefore)

    def ellipsRatio(self):
        """Calculates the ellipticity ratio of the object in the microscopy image by analyzing the major and minor axes of the detected contours."""
        image = self._image[int(self._image.shape[0]/2)]
        props = regionprops(label(image > ThresholdLegacy(nb_iteration=1000).getThreshold(image)))
        if not props:
            return 0.0
        largestRegion = max(props, key=lambda r: r.area)
        if largestRegion.axis_minor_length == 0:
            return 0.0
        self._ellipsRatio = largestRegion.axis_major_length / largestRegion.axis_minor_length
        self._orientation = np.rad2deg(largestRegion.orientation)%180

    def distance2D(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    

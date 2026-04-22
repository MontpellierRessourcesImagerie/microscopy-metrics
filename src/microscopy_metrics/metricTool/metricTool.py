import math
from matplotlib import image
import numpy as np

from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from skimage.measure import regionprops, label

from microscopy_metrics.utils import umToPx
from microscopy_metrics.thresholdTools.legacy import ThresholdLegacy
from microscopy_metrics.detectionTools.detection_tool import DetectionTool


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
        voxelVolume = self._pixelSize[0] * self._pixelSize[1] * self._pixelSize[2]
        imageFloat = self.setNormalizedImage(self._image)
        imageFloat = median_filter(imageFloat, size=5)
        thresholdAbs = ThresholdLegacy(nb_iteration=1000).getThreshold(imageFloat)
        binaryImage = imageFloat > thresholdAbs
        voxelCount = np.sum(binaryImage)
        volume = voxelCount * voxelVolume
        return volume

    def updateTile(self, tile, tileIndex):
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    tile[z][y][x] = (tileIndex & (1 << (z * 4 + y * 2 + x))) > 0

    def surfaceAreaLutD3(self):
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
        return sum(val << i for i, val in enumerate(configValues))

    def getHistogram3D(self):
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
        sum = 0
        for i in range(len(histogram)):
            sum += histogram[i] * lut[i]
        return sum

    def getSurface(self):
        lut = self.surfaceAreaLutD3()
        histo = self.getHistogram3D()
        return self.applyLut(histo, lut)

    def sphericity(self):
        c = 36 * math.pi
        volume = self.getVolume()
        surface = self.getSurface()
        
        if surface == 0:
            return 0.0
            
        print((c * (volume**2)) / (surface**3))
        self._sphericity = (c * (volume**2)) / (surface**3)

    def distance(self, x1, x2):
        return abs(x1 - x2)

    def multiplePeak(self):
        image = self._image[:,int(self._image.shape[1]/2),:]
        image = median_filter(image, size=3)
        XScore = 0.0
        TotalX = 0
        for x in range(image.shape[1]):
            profile = image[:,x]
            amp = float(np.max(profile) - np.min(profile))
            prominenceMin = amp * float(0.5)
            peaks, props = find_peaks(profile, prominence=prominenceMin, distance=3)
            if len(peaks) > 1 and not (any(peaks > self._image.shape[0]/2) and any(peaks < self._image.shape[0]/2)):
                maxPeak = peaks[np.argmax(profile[peaks])]
                if all(peaks > self._image.shape[0]/2):
                    print(f"X Profile {x} : {peaks} (To the right)")
                    XScore += 1 * (self.distance(x, self._image.shape[1]/2) / (self._image.shape[1])) * (profile[maxPeak] / amp) * (1/ (0.5*self.distance(peaks[0], peaks[-1])))
                else : 
                    print(f"X Profile {x} : {peaks} (To the left)")
                    XScore -= 1 * (self.distance(x, self._image.shape[1]/2) / (self._image.shape[1])) * (profile[maxPeak] / amp) * (1/ (0.5*self.distance(peaks[0], peaks[-1])))
                TotalX += 1
        if TotalX > 0:
            XScore /= TotalX
        image = self._image[:,:,int(self._image.shape[1]/2)]
        image = median_filter(image, size=3)
        YScore = 0.0
        TotalY = 0
        for y in range(image.shape[1]):
            profile = image[:,y]
            amp = float(np.max(profile) - np.min(profile))
            prominenceMin = amp * float(0.5)
            peaks, props = find_peaks(profile, prominence=prominenceMin, distance=3)
            if len(peaks) > 1 and not (any(peaks > self._image.shape[0]/2) and any(peaks < self._image.shape[0]/2)):
                maxPeak = peaks[np.argmax(profile[peaks])]
                if all(peaks > self._image.shape[0]/2):
                    print(f"Y Profile {y} : {peaks} (To the right)")
                    YScore += 1 * (self.distance(y, self._image.shape[1]/2) / (self._image.shape[1])) * (profile[maxPeak] / amp) * (1/ (0.5*self.distance(peaks[0], peaks[-1])))
                else : 
                    print(f"Y Profile {y} : {peaks} (To the left)")
                    YScore -= 1 * (self.distance(y, self._image.shape[1]/2) / (self._image.shape[1])) * (profile[maxPeak] / amp) * (1/ (0.5*self.distance(peaks[0], peaks[-1])))
                TotalY += 1
        if TotalY > 0:
            YScore /= TotalY
        Score = np.sqrt(XScore**2 + YScore**2)
        print(f"X Score : {XScore} | Y Score : {YScore} | Final Score : {Score}")
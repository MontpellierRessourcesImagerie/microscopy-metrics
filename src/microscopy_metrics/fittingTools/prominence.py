import numpy as np

from scipy.signal import find_peaks

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.utils import pxToUm, umToPx


class Prominence(FittingTool):
    """Class for fitting a 1D Gaussian curve to the PSF profile of a microscopy image based on the prominence of peaks in the intensity profile.
    This class inherits from the FittingTool base class and implements methods specific to fitting based on peak prominence.
    """

    name = "Prominence"

    def __init__(self):
        super().__init__()
        self._prominenceRel = 0.1

    def processSingleFit(self, index):
        """Processes a single fit by analyzing the intensity profiles along the Z, Y, and X axes, identifying peaks based on their prominence, and calculating the full width at half maximum (FWHM) for the detected peaks.
         The method retrieves the local centroid of the image, extracts the intensity profiles along the three axes, and applies the find_peaks function to identify peaks based on their prominence. For each detected peak, the method calculates the FWHM by determining the points where the intensity crosses half of the peaks prominence.
         The calculated FWHM values and corresponding parameters are stored in the class attributes for further analysis and evaluation.
        Args:
            index (int): The index of the fit being processed, used for storing results in the parameters attribute.
        Returns:
            list: A list containing the index, calculated FWHM values, covariance matrix, coefficient of determination, and parameters for the fit.
        """

        def cross(a, b):
            if a < 0 or b < 0 or a >= len(profile) or b >= len(profile):
                return float(a) if 0 <= a < len(profile) else float(b)
            v0, v1 = profile[a], profile[b]
            return a + (h - v0) * (b - a) / (v1 - v0) if v0 != v1 else float(a)

        if self._image is None:
            return None
        if self._centroid is None:
            return None
        physic = self.getLocalCentroid()
        profiles = [
            self._image[:, physic[1], physic[2]],
            self._image[physic[0], :, physic[2]],
            self._image[physic[0], physic[1], :],
        ]
        for idx, profile in enumerate(profiles):
            amp = float(np.max(profile) - np.min(profile))
            prominenceMin = amp * float(self._prominenceRel)
            peaks, props = find_peaks(profile, prominence=prominenceMin)
            if not peaks.size:
                return None
            i = np.argmax(props["prominences"])
            pk = peaks[i]
            h = profile[pk] - props["prominences"][i] / 2.0
            above = np.where(profile > h)[0]
            if len(above) < 2:
                return None
            lIdx, rIdx = above[0], above[-1]
            if lIdx == 0 or rIdx >= len(profile) - 1:
                return None
            leftCrossing = cross(lIdx - 1, lIdx)
            rightCrossing = cross(rIdx, rIdx + 1)
            self.fwhms[idx] = pxToUm(rightCrossing - leftCrossing, self._spacing[idx])
            self.parameters[0] += amp / 3.0
            self.parameters[1] += float(np.min(profile)) / 3.0
            self.parameters[2 + idx] = float(pk)
            self.parameters[5 + idx] = umToPx(self.fwhms[idx], self._spacing[idx]) / (
                2 * np.sqrt(2 * np.log(2))
            )
        return [
            index,
            self.fwhms,
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
            ],
            self.determinations,
            self.parameters,
        ]

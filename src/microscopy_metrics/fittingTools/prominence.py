import os
import numpy as np
from scipy.signal import find_peaks
from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.utils import pxToUm


class Prominence(FittingTool):
    name = "Prominence"
    def __init__(self):
        super().__init__()
        self._prominenceRel = 0.1

    def processSingleFit(self,index):
        def cross(a, b):
            if a < 0 or b < 0 or a >= len(profile) or b >= len(profile):
                return float(a) if 0 <= a < len(profile) else float(b)
            v0, v1 = profile[a], profile[b]
            return a + (h - v0) * (b - a) / (v1 - v0) if v0 != v1 else float(a)
        
        if self._image is None : 
            return None
        if self._centroid is None :
            return None
        physic = self.getLocalCentroid()
        profiles = [
            self._image[:,physic[1],physic[2]],
            self._image[physic[0],:,physic[2]],
            self._image[physic[0],physic[1],:],
        ]
        fwhms = []
        for i,profile in enumerate(profiles) :
            amp = float(np.max(profile) - np.min(profile))
            prominenceMin = amp * float(self._prominenceRel)
            peaks,props = find_peaks(profile, prominence=prominenceMin)
            if not peaks.size:
                return None
            i = np.argmax(props['prominences'])
            pk = peaks[i]
            h = profile[pk] - props['prominences'][i] / 2.0
            above = np.where(profile > h)[0]
            if len(above) < 2 :
                return None
            lIdx,rIdx = above[0], above[-1]
            if lIdx == 0 or rIdx >= len(profile) - 1 :
                return None

            
            leftCrossing = cross(lIdx - 1, lIdx)
            rightCrossing = cross(rIdx, rIdx + 1)
            fwhms.append(pxToUm(rightCrossing - leftCrossing,self._spacing[i]))
        return [index,fwhms, [[0.0000,0.0000,0.0000,0.0000],[0.0000,0.0000,0.0000,0.0000],[0.0000,0.0000,0.0000,0.0000]], [0.0000,0.0000,0.0000], [0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000]]


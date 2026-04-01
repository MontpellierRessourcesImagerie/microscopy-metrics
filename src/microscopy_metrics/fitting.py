from os import sched_get_priority_max

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from microscopy_metrics.fittingTools.fittingTool import FittingTool

class Fitting(object):
    def __init__(self):
        self._images = []
        self._centroids = []
        self._spacing = [1, 1, 1]
        self._rois = []
        self._outputDir = ""
        self.results = []
        self.fitType = "1D"
        self._thresholdRSquared = 0.95
        self.retainedId = []
        self._prominenceRel = None

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if len(images) == 0 or images is None:
            raise ValueError("Please, send at list one image")
        self._images = images

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, centroids):
        if len(centroids) == 0 or centroids is None:
            raise ValueError("Please, send at list one centroid")
        self._centroids = centroids

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        if value is None or len(value) == 0:
            raise ValueError("Shape format not compatible with current image")
        self._spacing = value

    @property
    def rois(self):
        return self._rois

    @rois.setter
    def rois(self, rois):
        if len(rois) == 0 or rois is None:
            raise ValueError("Please, send at list one ROI")
        self._rois = rois

    @property
    def outputDir(self):
        return self._outputDir

    @outputDir.setter
    def outputDir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The outputDir is wrong")
        self._outputDir = value


    def runFitting(self, index):
        fitTool = FittingTool.getInstance(self.fitType)
        fitTool._image = self._images[index]
        fitTool._centroid = self._centroids[index]
        fitTool._spacing = self.spacing
        fitTool._roi = self._rois[index]
        fitTool._outputDir = self._outputDir
        if hasattr(fitTool,"_prominenceRel") and self._prominenceRel is not None:
            fitTool._prominenceRel = self._prominenceRel
        return fitTool.processSingleFit(index)

    def computeFitting(self):
        self.results = []
        workers = int(os.cpu_count() * 0.75)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.runFitting, i): i
                for i, roi in enumerate(self._rois)
            }

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
        tmp = []
        self.retainedId = []
        for i,result in enumerate(self.results) :
            if result is None : 
                print(f"Bead {i} is None")
                continue
            meanDetermination = (result[3][0] + result[3][1] + result[3][2])/3.0
            if meanDetermination >= self._thresholdRSquared : 
                tmp.append(result)
                self.retainedId.append(result[0])
        if len(tmp) > 0 :
            self.results = tmp

            



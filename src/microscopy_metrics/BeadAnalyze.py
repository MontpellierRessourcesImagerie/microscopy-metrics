
class BeadAnalyze(object):
    def __init__(self, id = 0, image = None, roi = None, centroid = None):
        self._id = id
        self._image = image
        self._roi = roi
        self._centroid = centroid
        self._rejected = False
        self._rejectionDesc = ""
        self._metricTool = None
        self._fitTool = None
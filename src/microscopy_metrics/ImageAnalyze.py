
class ImageAnalyze(object):
    def __init__(self, path = "~/", BeadAnalyze = [], _beadSize = 1.0, pixelSize = [1,1,1]):
        self._path = path
        self._beadAnalyze = BeadAnalyze
        self._beadSize = _beadSize
        self._pixelSize = pixelSize
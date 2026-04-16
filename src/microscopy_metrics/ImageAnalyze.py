
class ImageAnalyze(object):
    def __init__(self, image = None, path = "~/", BeadAnalyze = None, beadSize = 1.0, pixelSize = [1,1,1]):
        self._path = path
        self._beadAnalyze = BeadAnalyze if BeadAnalyze is not None else []
        self._beadSize = beadSize 
        self._pixelSize = pixelSize
        self._image = image
        self._theoreticalResolution = [0.0, 0.0, 0.0]
        self._meanSBR = 0.0
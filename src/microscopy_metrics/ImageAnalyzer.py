class ImageAnalyzer(object):
    """Class for managing the image data, bead analysis results, and various parameters such as bead size and pixel size."""

    def __init__(
        self, image=None, path="~/", BeadAnalyzer=None, beadSize=1.0, pixelSize=[1, 1, 1]
    ):
        self._path = path
        self._beadAnalyzer = BeadAnalyzer if BeadAnalyzer is not None else []
        self._beadSize = beadSize
        self._pixelSize = pixelSize
        self._image = image
        self._theoreticalResolution = [0.0, 0.0, 0.0]
        self._meanSBR = 0.0
        self._meanComaticity = 0.0
        self._meanSphericalAberration = 0.0
        self._meanAstigmatism = 0.0
        self._meanContrast = 0.0
        self._meanEllipsRatio = 0.0
        self._meanOrientation = 0.0

    def toDict(self):
        """Converts the image analysis results into a dictionary format for easier access and manipulation of the image's data.
        Returns:
            dict: A dictionary containing the image's analysis results.
        """
        return {
            "path": self._path,
            "beadAnalyzer": [bead._id for bead in self._beadAnalyzer],
            "beadSize": self._beadSize,
            "pixelSize": self._pixelSize,
            "theoreticalResolution": self._theoreticalResolution,
            "meanSBR": self._meanSBR,
            "meanComaticity": self._meanComaticity,
            "meanSphericalAberration": self._meanSphericalAberration,
            "meanEllipsRatio": self._meanEllipsRatio,
            "meanOrientation": self._meanOrientation,
        }

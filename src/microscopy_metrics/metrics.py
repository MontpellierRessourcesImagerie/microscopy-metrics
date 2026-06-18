import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed


class Metrics(object):
    """Class for calculating various metrics related to microscopy images."""

    def __init__(self):
        self._imageAnalyzer = None
        self._ringInnerDistance = None
        self._ringThickness = None

        self._TheoreticalResolutionTool = None

    @property
    def ringInnerDistance(self):
        return self._ringInnerDistance

    @ringInnerDistance.setter
    def ringInnerDistance(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringInnerDistance = value

    @property
    def ringThickness(self):
        return self._ringThickness

    @ringThickness.setter
    def ringThickness(self, value):
        if not isinstance(value, float):
            raise ValueError("Please, enter a float value as ringInnerDistance")
        self._ringThickness = value

    @property
    def theoreticalResolutionTool(self):
        return self._TheoreticalResolutionTool

    @theoreticalResolutionTool.setter
    def theoreticalResolutionTool(self, value):
        self._TheoreticalResolutionTool = value

    def signalToBackgroundRatioRing(self):
        """Calculates the signal-to-background ratio (SBR) for a set of microscopy images using a ring-based method.
        It uses a ThreadPoolExecutor to parallelize the processing of multiple images for improved performance.
        Raises:
            ValueError: If there are no images in the input list or if any of the images have an incorrect format.
        """
        self._imageAnalyzer._meanSBR = 0.0

        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    bead.runSBRMetric,
                    self._imageAnalyzer._pixelSize,
                    self._ringInnerDistance,
                    self._ringThickness,
                ): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                _ = future.result()
        total = 0
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                self._imageAnalyzer._meanSBR += bead._metricTool._SBR
                total += 1
        self._imageAnalyzer._meanSBR = self._imageAnalyzer._meanSBR / total if total > 0 else 0

    def runPrefittingMetrics(self):
        """Runs the pre-fitting metrics calculations, including signal-to-background ratio (SBR) calculation and theoretical resolution estimation."""
        self.SBR = []
        yield {"desc": "SBR calculation..."}
        self.signalToBackgroundRatioRing()
        yield {"desc": "Estimating theoretical resolution..."}
        self._imageAnalyzer._theoreticalResolution = (
            self._TheoreticalResolutionTool.getTheoreticalResolution()
        )
        self._imageAnalyzer._samplingDistance = self._TheoreticalResolutionTool.getSamplingDistance()
        yield {"desc": "Mesh-based metrics calculation..."}
        self.runMeshMetrics()

    def runMeshMetrics(self):
        """Runs the mesh-based metrics calculations, including mesh building, concavity, curvature, and sphericity calculations."""
        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    bead._metricTool.meshMetrics,
                ): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                _ = future.result()

    def runSingleMetrics(self, bead):
        """Runs the complete metrics calculation process for a single index, including pre-fitting metrics and fitting metrics calculations."""
        if bead._rejected == False and bead._roi is not None:
            FWHM = bead._fitTool.fwhms
            bead._metricTool.lateralAsymmetryRatio(FWHM)
            bead._metricTool.sphericity()
            bead._metricTool.comaticity()
            bead._metricTool.sphericalAberration()
            bead._metricTool.astigmatism(bead._fitTool.getMu(), bead._fitTool.getSigma())
            bead._fitTool.computeContrast()
            bead._metricTool.ellipsRatio()
            bead._metricTool.skeletonizePath()

    def runMetrics(self):
        """Runs the complete metrics calculation process, including pre-fitting metrics and fitting metrics calculations."""
        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        yield {"desc": "Running metrics calculation..."}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.runSingleMetrics, bead): i
                for i, bead in enumerate(self._imageAnalyzer._beadAnalyzer)
                if bead._rejected == False and bead._roi is not None
            }
            for future in as_completed(futures):
                _ = future.result()
        yield {"desc": "Calculating mean metrics..."}
        self._imageAnalyzer._meanComaticity = np.mean([bead._metricTool._comaticity for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanSphericalAberration = np.mean([bead._metricTool._sphericalAberration for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanAstigmatism = np.mean([bead._metricTool._astigmatism for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanContrast = np.mean([bead._fitTool.contrast for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanEllipsRatio = np.mean([bead._metricTool._ellipsRatio for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanOrientation = np.mean([bead._metricTool._orientation for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanSkeleton2Extremities = np.mean([bead._metricTool._skeleton2Extremities for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanRMin = np.mean([bead._metricTool._RMin for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanLAR = np.mean([bead._metricTool._LAR for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanSphericity = np.mean([bead._metricTool._sphericity for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])
        self._imageAnalyzer._meanConcavity = np.mean([bead._metricTool.meshBuilder._concavity for bead in self._imageAnalyzer._beadAnalyzer if bead._rejected == False])

    def calculateDensity(self):
        """Calculates the density of beads in the microscopy images based on their centroid coordinates."""
        if len(self._imageAnalyzer._beadAnalyzer) == 0:
            raise ValueError("You must have at least one PSF")
        YProjImage = np.zeros(self._imageAnalyzer._image.shape[1])
        XProjImage = np.zeros(self._imageAnalyzer._image.shape[2])
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                for y in range(bead._roi[0][1], bead._roi[2][1]):
                    YProjImage[y] = 1
                for x in range(bead._roi[0][2], bead._roi[2][2]):
                    XProjImage[x] = 1
        densityY = np.sum(YProjImage) / self._imageAnalyzer._image.shape[1]
        densityX = np.sum(XProjImage) / self._imageAnalyzer._image.shape[2]
        density = (densityY + densityX) / 2 if (densityY > 0 and densityX > 0) else 0
        self._imageAnalyzer._density = density

    def GenerateHeatmap(self, outputDir=None):
        """Generates a heatmap visualization of the signal-to-background ratio (SBR) for the microscopy images, providing insights into the spatial distribution of SBR across the images."""
        self.calculateDensity()
        if self._imageAnalyzer._density < 0.5:
            print(f"Density of beads is too low for heatmap generation. Generating placeholder image...")
        xCoords = []
        yCoords = []
        sbrValues = []
        ComaticityValues = []
        SphericalAberrationValues = []
        AstigmatismValues = []
        ContrastValues = []
        EllipsRatioValues = []
        OrientationValues = []
        RMinValues = []
        Skeleton2ExtremitiesValues = []
        LARValues = []
        SphericityValues = []
        for bead in self._imageAnalyzer._beadAnalyzer:
            if bead._rejected == False and bead._roi is not None:
                yCoords.append(bead._centroid[1])
                xCoords.append(bead._centroid[2])
                sbrValues.append(bead._metricTool._SBR)
                ComaticityValues.append(bead._metricTool._comaticity)
                SphericalAberrationValues.append(bead._metricTool._sphericalAberration)
                AstigmatismValues.append(bead._metricTool._astigmatism)
                ContrastValues.append(bead._fitTool.contrast)
                EllipsRatioValues.append(bead._metricTool._ellipsRatio)
                OrientationValues.append(bead._metricTool._orientation)
                RMinValues.append(bead._metricTool._RMin)
                Skeleton2ExtremitiesValues.append(bead._metricTool._skeleton2Extremities)
                LARValues.append(bead._metricTool._LAR)
                SphericityValues.append(bead._metricTool._sphericity)
                bead._metricTool.generateBeadOrientation(os.path.join(outputDir, f"bead_{bead._id}"))
        self.HeatmapGenerator(outputDir, sbrValues, xCoords, yCoords, MetricName="SBR")
        self.HeatmapGenerator(outputDir, ComaticityValues, xCoords, yCoords, MetricName="Comaticity")
        self.HeatmapGenerator(outputDir, SphericalAberrationValues, xCoords, yCoords, MetricName="SphericalAberration")
        self.HeatmapGenerator(outputDir, AstigmatismValues, xCoords, yCoords, MetricName="Astigmatism")
        self.HeatmapGenerator(outputDir, ContrastValues, xCoords, yCoords, MetricName="Contrast")
        self.HeatmapGenerator(outputDir, EllipsRatioValues, xCoords, yCoords, MetricName="EllipsRatio")
        self.HeatmapGenerator(outputDir, OrientationValues, xCoords, yCoords, MetricName="Orientation")
        self.HeatmapGenerator(outputDir, RMinValues, xCoords, yCoords, MetricName="RMin")
        self.HeatmapGenerator(outputDir, Skeleton2ExtremitiesValues, xCoords, yCoords, MetricName="Skeleton2Extremities")
        self.HeatmapGenerator(outputDir, LARValues, xCoords, yCoords, MetricName="LAR")
        self.HeatmapGenerator(outputDir, SphericityValues, xCoords, yCoords, MetricName="Sphericity")
        

    def HeatmapPlaceholder(self, outputDir=None, MetricName="SBR"):
        """Returns a white placeholder image with centered text."""
        image = self._imageAnalyzer._image
        if image.ndim == 3:
            image = np.max(image, axis=0)
        height, width = image.shape
        text = f"{MetricName} Heatmap is not available : not enough beads."
        fig, ax = plt.subplots(figsize=(10, 8))
        grey_image = np.full((height, width), 0.5, dtype=float)
        ax.imshow(grey_image, cmap="gray", vmin=0, vmax=1)
        ax.text(
            width / 2,
            height / 2,
            text,
            ha="center",
            va="center",
            fontsize=14,
            color="black",
            wrap=True,
        )
        ax.set_title(f"Interpolated {MetricName} Heatmap", fontsize=14)
        ax.axis("off")
        if outputDir is not None:
            plt.savefig(
                os.path.join(outputDir, f"{MetricName}_Heatmap.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()
        return fig
    
    def HeatmapGenerator(self,outputDir,Values,xCoords,yCoords,MetricName="SBR"):
        if len(Values) == 0:
            print(f"No values for {MetricName} heatmap generation. Generating placeholder image.")
            return self.HeatmapPlaceholder(outputDir, MetricName)
        if self._imageAnalyzer._density < 0.5:
            print(f"Density of beads is too low ({self._imageAnalyzer._density:.4f}) for {MetricName} heatmap generation. Generating placeholder image.")
            return self.HeatmapPlaceholder(outputDir, MetricName)
        image = self._imageAnalyzer._image
        if image.ndim == 3:
            image = np.max(image, axis=0)
        fig,ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image, cmap="gray")
        height, width = image.shape
        gridX, gridY = np.meshgrid(np.arange(width), np.arange(height))
        points = np.column_stack((xCoords, yCoords))
        method = 'cubic' if len(Values) >= 4 else 'nearest'
        heatmap = griddata(points, Values, (gridX, gridY), method=method)
        if np.any(np.isnan(heatmap)):
            heatmapNearest = griddata(points, Values, (gridX, gridY), method='nearest')
            heatmap[np.isnan(heatmap)] = heatmapNearest[np.isnan(heatmap)]
        heatmap = gaussian_filter(heatmap, sigma=15)
        im = ax.imshow(heatmap, cmap="plasma", alpha=0.6)
        ax.scatter(xCoords, yCoords, c='black', s=20, marker='o', alpha=0.8)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{MetricName}", fontsize=12)
        ax.set_title(f"Interpolated {MetricName} Heatmap", fontsize=14)
        ax.axis("off")
        if outputDir is not None:
            plt.savefig(os.path.join(outputDir, f"{MetricName}_Heatmap.png"), bbox_inches="tight", dpi=300)
        plt.close()
        return fig

How To
=======

This project is developed in parallel with a **Napari plugin** for interactive visualization and analysis of microscopy images.
You can use it in two ways:

1. **With the Napari GUI** (recommended for interactive analysis):
   
   Install the plugin via pip:

   .. code-block:: bash

      pip install napari-microscopy-metrics

2. **As a standalone Python library** (for programmatic analysis):
   
   Use the modules directly in your Python scripts, as described below.

Here is an overview of the available features:

- :ref:`psf_generation`
- :ref:`bead_detection`
- :ref:`pre_fitting_metrics`
- :ref:`fwhm_estimation`
- :ref:`metrics_calculation`
- :ref:`report_generation`

---

.. _psf_generation:

=====================
PSF Generation
=====================

The **PSF (Point Spread Function) generation module** allows you to create synthetic PSFs with or without aberrations for testing and analysis.

---------------------
Random PSF Generation
---------------------

.. code-block:: python

   from microscopy_metrics.scripts.PSFGenerator.PSF import PSFRandomParameter

   # Set the output directory for results
   path = "path/to/your/result/folder"

   # Generate a PSF without aberration (random parameters)
   psf = PSFRandomParameter()

-----------
Aberrations
-----------

You can generate PSFs with specific aberrations using the ``aberrationType`` parameter:

.. code-block:: python

    # Comatic aberration
    psf = PSFRandomParameter(aberrationType="comatic")

    # Astigmatism aberration
    psf = PSFRandomParameter(aberrationType="astigmatism")

    # Spherical aberration
    psf = PSFRandomParameter(aberrationType="spherical")

.. note::
   - Aberrations are applied **randomly** by default.

--------------------
Fixed PSF Generation
--------------------

You can also generate PSFs with **fixed parameters** using the ``PSFGenerator`` class:

.. code-block:: python

    from microscopy_metrics.scripts.PSFGenerator.PSF import PSFGenerator

    # Generate a PSF with fixed parameters
    psf = PSFGenerator()

---

.. _bead_detection:

=====================
Bead Detection
=====================

The **bead detection module** identifies beads in microscopy images and extracts their **regions of interest (ROIs)**.
This step is required before calculating metrics or fitting PSFs.

-------------
Basic Usage
-------------

.. code-block:: python

   import numpy as np
   from microscopy_metrics.detection import Detection
   from microscopy_metrics.thresholdTools.threshold_tool import Threshold
   from microscopy_metrics.detectionTools.detection_tool import DetectionTool

   # Initialize the detector
   detector = Detection()
   detector.image = psf.psf  # Assign the PSF image

   # Configure the detection tool (e.g., "Centroids", "peak local maxima", etc.)
   detectionTool = DetectionTool.getInstance("Centroids")
   detectionTool._image = psf.psf
   detectionTool._thresholdTool = Threshold.getInstance("legacy")

   # Assign the detection tool to the detector
   detector._detectionTool = detectionTool

   # Set detection parameters
   detector.cropFactor = 5          # Crop factor for ROI extraction
   detector.beadSize = 0.2         # Expected bead size (in micrometers)
   detector.rejectionDistance = 0.5 # Minimum distance between beads (in micrometers)
   detector.pixelSize = np.array([psf.dz, psf.dxy, psf.dxy])  # Pixel size (Z, Y, X)

   # Run detection and save ROIs
   for _ in detector.run(outputDir=path):
       pass

   # Access the image analyzer for further processing
   imageAnalyzer = detector._imageAnalyzer
   imageAnalyzer._path = path

.. note::
   - The ``cropFactor`` and ``beadSize`` determines how much to crop around each detected bead.
   - The ``rejectionDistance`` help filter top and bottom beads.

---

.. _pre_fitting_metrics:

=========================
Pre-Fitting Metrics
=========================

Before estimating the **Full Width at Half Maximum (FWHM)** of the PSF, you can compute **theoretical metrics** such as:
    - **Theoretical resolution** (based on microscope parameters).
    - **Signal-to-background ratio (SBR)**.

------------
Basic Usage
------------

.. code-block:: python

   import os
   from microscopy_metrics.metrics import Metrics
   from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution

   # Initialize the metrics tool
   MetricTool = Metrics()
   MetricTool._imageAnalyzer = imageAnalyzer

   # Configure theoretical resolution parameters
   MetricTool._TheoreticalResolutionTool = TheoreticalResolution.getInstance("widefield")
   MetricTool._TheoreticalResolutionTool._numericalAperture = psf.NA
   MetricTool._TheoreticalResolutionTool._emissionWavelength = psf.wvl
   MetricTool._TheoreticalResolutionTool._refractiveIndex = psf.ni
   MetricTool._TheoreticalResolutionTool._excitationWavelength = 0.225  # Default excitation wavelength

   # Set ring parameters for SBR calculation
   MetricTool._ringInnerDistance = 1.0  # Inner radius for background estimation
   MetricTool._ringThickness = 2.0      # Thickness of the background ring

   # Run pre-fitting metrics
   for _ in MetricTool.runPrefittingMetrics():
       pass

   # Save mesh data for each bead (optional)
   for bead in imageAnalyzer._beadAnalyzer:
       if bead._rejected == False and bead._roi is not None and bead._metricTool.meshBuilder is not None:
           bead._metricTool.meshBuilder.saveMesh(os.path.join(path, f"bead_{bead._id}_mesh.obj"))

.. note::
   - The **theoretical resolution** is calculated based on the **numerical aperture (NA)**, **wavelength**, and **refractive index**.
   - The **signal-to-background ratio** helps assess the quality of the detected beads.

---

.. _fwhm_estimation:

=====================
FWHM Estimation
=====================

Estimate the **Full Width at Half Maximum (FWHM)** of the PSF by fitting a **Gaussian function** to each detected bead.
This step is crucial for characterizing the **resolution** of your microscope.

-----------
Basic Usage
-----------

.. code-block:: python

   from microscopy_metrics.fitting import Fitting

   # Initialize the fitting tool
   FittingTool = Fitting()
   FittingTool._imageAnalyzer = imageAnalyzer

   # Configure fitting parameters
   FittingTool.fitType = "2D"  # or "1D", "3D", "2D Ellips", "2D Rotation", "3D Rotation"
   FittingTool._thresholdRSquared = 0.5  # Minimum R² value for a valid fit

   # Run the fitting
   FittingTool.computeFitting()

.. note::
   - The ``fitType`` determines the **dimensionality and model** used for fitting (e.g., 2D Gaussian, 3D Gaussian with rotation).
   - The ``_thresholdRSquared`` filters out poor fits (adjust based on your data quality).

---

.. _metrics_calculation:

=====================
Metrics Calculation
=====================

After fitting, compute **advanced metrics** such as:
    - **Asymmetry** (deviation from a perfect Gaussian).
    - **Aberration metrics** (e.g., astigmatism, coma).

------------
Basic Usage
------------

.. code-block:: python

   # Assign the fitting results to the metrics tool
   MetricTool._imageAnalyzer = FittingTool._imageAnalyzer

   # Run metrics calculation
   for _ in MetricTool.runMetrics():
       pass

   # Access the updated image analyzer
   imageAnalyzer = MetricTool._imageAnalyzer

.. note::
   - Metrics are calculated **per bead** and stored in ``imageAnalyzer._beadAnalyzer``.
   - You can access individual bead metrics (e.g., FWHM, SNR) via ``bead._metricTool``.

---

.. _report_generation:

=====================
Report Generation
=====================

Generate **reports** summarizing the analysis results, including:
    - **Acquisition parameters** (e.g., microscope settings).
    - **Detected beads** (positions, intensities).
    - **Fitted PSFs** (FWHM, R² values).
    - **Metrics** (resolution, SNR, asymmetry).
    - **Visualizations** (heatmaps, fitting curves).

------------
Basic Usage
------------

.. code-block:: python

   from microscopy_metrics.report_generator import ReportGenerator

   # Crop PSF images for the report
   detector.cropPsf(path)          # Crop individual PSFs
   detector.GlobalCropPsf(path)    # Crop global PSF overview

   # Generate heatmaps
   MetricTool.GenerateHeatmap(path)

   # Display fitting results
   FittingTool.displayFitting(path)

   # Generate the HTML report
   ReportGenerator = ReportGenerator().getInstance("HTML")
   ReportGenerator._inputDir = path
   ReportGenerator._imageAnalyzer = imageAnalyzer
   ReportGenerator.generateReport(path)

.. note::
   - Reports are saved in the specified ``path`` directory.
   - The **HTML report** includes interactive visualizations (if viewed in a browser).
   - There also exists a **PDF report** option, which can be generated by changing the report type in ``ReportGenerator.getInstance("PDF")`` and a CSV report can be generated by changing the report type in ``ReportGenerator.getInstance("CSV")``.
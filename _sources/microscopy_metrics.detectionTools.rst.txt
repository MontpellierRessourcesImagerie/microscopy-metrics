microscopy_metrics.detectionTools package
=========================================

This package contains the detection tools used for detecting objects in microscopy images. The detection tools are implemented as classes that inherit from the base class `DetectionTool`. Each detection tool has its own unique algorithm for detecting objects in images.

DetectionTool
-------------

.. autoclass:: microscopy_metrics.detectionTools.detection_tool.DetectionTool
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detectionTools.detection_tool.DetectionTool`

----

PeakLocalMaxDetector
--------------------

.. autoclass:: microscopy_metrics.detectionTools.detection_tool.PeakLocalMaxDetector
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detectionTools.detection_tool.PeakLocalMaxDetector`

----

CentroidDetector
-----------------

.. autoclass:: microscopy_metrics.detectionTools.detection_tool.CentroidDetector
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detectionTools.detection_tool.CentroidDetector`

----

BlobLogDetector
----------------

.. autoclass:: microscopy_metrics.detectionTools.detection_tool.BlobLogDetector
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detectionTools.detection_tool.BlobLogDetector`

----

BlobDogDetector
----------------

.. autoclass:: microscopy_metrics.detectionTools.detection_tool.BlobDogDetector
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detectionTools.detection_tool.BlobDogDetector`


Detection 
----------

This module uses the detection tools to detect objects in microscopy images. It provides a unified interface for using different detection tools and allows for easy switching between them.

.. automodule:: microscopy_metrics.detection
   :members:
   :show-inheritance:
   :noindex:

   **Source Code:** :py:mod:`microscopy_metrics.detection`
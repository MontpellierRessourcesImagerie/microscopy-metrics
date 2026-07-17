
import os
import shutil

from datetime import datetime
from skimage.io import imread
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle

from microscopy_metrics.metrics import Metrics
from microscopy_metrics.fitting import Fitting
from microscopy_metrics.detection import Detection
from microscopy_metrics.ImageAnalyzer import ImageAnalyzer
from microscopy_metrics.report_generator import ReportGenerator
from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.thresholdTools.threshold_tool import Threshold
from microscopy_metrics.detectionTools.detection_tool import DetectionTool
from microscopy_metrics.resolutionTools.theoretical_resolution import TheoreticalResolution



class BatchAnalyzer(object):

    _detectionMethod = "peak local maxima"
    _MinDistance = 5
    _Sigma = 2.0
    _thresholdMethod = "Otsu"
    _relThreshold = 0.1
    _TheoreticalBeadSize = 0.2
    _ZRejectionMargin = 0.5
    _cropFactor = 1.0
    _thresholdIntensity = 0.1
    _pixelSize = [0.1, 0.1, 0.1]
    _prominenceDoublePass = 0.5
    _annulusInnerDistance = 0.5
    _annulusThickness = 0.2
    _MicroscopeType = "widefield"
    _numericalAperture = 1.4
    _emissionWavelength = 520.0
    _refractionIndex = 1.518
    _excitationWavelength = 488.0
    _FitType = "1D"
    _prominenceRel = 0.1
    _thresholdRSquared = 0.8
    _listReports = ["PDF", "HTML", "CSV"]
    _detectionDatas = {}
    _thresholdDatas = {}
    _roiDatas = {}
    _fittingDatas = {}
    _microscopeDatas = {}

    _imageAnalzed = []

    def __init__(self, folderPath:str):
        if os.path.isdir(folderPath):
            self._folderPath = folderPath
        else:
            raise ValueError(f"{folderPath} is not a valid directory path")
        self._imagePaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and f.lower().endswith(('.tif', '.tiff'))]
        if not self._imagePaths:
            raise ValueError(f"No .tif or .tiff files found in {folderPath}")
        
    def createDetectionTool(self,imagePath: str):
        """Creates a DetectionTool instance based on the specified detection method and parameters.
        
        Returns:
            DetectionTool: An instance of the DetectionTool class configured with the specified parameters.
        """
        image = imread(imagePath)
        detectionTool = Detection()
        detectionTool._image = image
        detectionTool._detectionTool = DetectionTool().getInstance(self._detectionMethod)
        if detectionTool._detectionTool is not None:
            if hasattr(detectionTool._detectionTool, "_minDistance"):
                detectionTool._detectionTool._minDistance = self._MinDistance
            if hasattr(detectionTool._detectionTool, "_sigma"):
                detectionTool._detectionTool._sigma = self._Sigma
        else:
            raise ValueError(f"Detection method '{self._detectionMethod}' is not recognized.")
        detectionTool._detectionTool._thresholdTool = Threshold.getInstance(self._thresholdMethod)
        if detectionTool._detectionTool._thresholdTool is not None and hasattr(detectionTool._detectionTool._thresholdTool, "_relThreshold"):
            detectionTool._detectionTool._relThreshold = self._relThreshold
        detectionTool._beadSize = self._TheoreticalBeadSize
        detectionTool._rejectionDistance = self._ZRejectionMargin
        detectionTool._cropFactor = self._cropFactor
        detectionTool._thresholdIntensity = self._thresholdIntensity
        detectionTool._pixelSize = self._pixelSize
        detectionTool._prominenceRel = self._prominenceDoublePass
        return detectionTool
    
    def createMetricTool(self, imageAnalyzer: ImageAnalyzer):
        """Creates a MetricTool instance based on the specified image analyzer.
        
        Args:
            imageAnalyzer (ImageAnalyzer): An instance of the ImageAnalyzer class containing the analyzed image data.
        
        Returns:
            MetricTool: An instance of the MetricTool class configured with the specified image analyzer.
        """
        MetricTool = Metrics()
        MetricTool._imageAnalyzer = imageAnalyzer
        MetricTool._ringInnerDistance = self._annulusInnerDistance
        MetricTool._ringThickness = self._annulusThickness
        MetricTool._TheoreticalResolutionTool = TheoreticalResolution.getInstance(self._MicroscopeType)
        MetricTool._TheoreticalResolutionTool._numericalAperture = self._numericalAperture
        MetricTool._TheoreticalResolutionTool._emissionWavelength = self._emissionWavelength / 1000
        MetricTool._TheoreticalResolutionTool._refractiveIndex = self._refractionIndex
        MetricTool._TheoreticalResolutionTool._excitationWavelength = self._excitationWavelength / 1000
        return MetricTool
    
    def createFittingTool(self, imageAnalyzer: ImageAnalyzer):
        """Creates a FittingTool instance based on the specified image analyzer.
        
        Args:
            imageAnalyzer (ImageAnalyzer): An instance of the ImageAnalyzer class containing the analyzed image data.
        
        Returns:
            FittingTool: An instance of the FittingTool class configured with the specified image analyzer.
        """
        FittingTool = Fitting()
        FittingTool.fitType = self._FitType
        FittingTool._prominenceRel = self._prominenceRel
        FittingTool._thresholdRSquared = self._thresholdRSquared
        FittingTool._imageAnalyzer = imageAnalyzer
        return FittingTool
    
    def generateReports(self, imageAnalyzer: ImageAnalyzer, outputDir: str):
        """Generates reports based on the specified image analyzer and saves them to the specified output directory.
        
        Args:
            imageAnalyzer (ImageAnalyzer): An instance of the ImageAnalyzer class containing the analyzed image data.
            outputDir (str): Path to the folder where the reports will be saved.
        """
        for reportType in self._listReports:
            PDFGenerator = ReportGenerator().getInstance(reportType)
            PDFGenerator._imageAnalyzer = imageAnalyzer
            PDFGenerator._inputDir = outputDir
            PDFGenerator._detectionDatas = self._detectionDatas
            PDFGenerator._thresholdDatas = self._thresholdDatas
            PDFGenerator._roiDatas = self._roiDatas
            PDFGenerator._fittingDatas = self._fittingDatas
            PDFGenerator._microscopeDatas = self._microscopeDatas
            PDFGenerator.generateReport(outputDir)

    def getActivePath(self,index, outputDir):
        """Returns the path of the image at the specified index in the list of image paths.
        
        Args:
            index (int): Index of the image path to retrieve.
            outputDir (str): Path to the output directory.
        """
        activePath = os.path.join(outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath
    
    def analyze(self):
        for imagePath in self._imagePaths:
            print(f"Analyzing {imagePath}...")
            outputDir = os.path.join(os.path.dirname(imagePath), os.path.basename(imagePath) + "_analysis")
            if os.path.exists(outputDir):
                shutil.rmtree(outputDir)
            os.makedirs(outputDir)
            DetectionTool = self.createDetectionTool(imagePath)
            for _ in DetectionTool.run(outputDir=outputDir):
                pass
            if DetectionTool._imageAnalyzer is None:
                print(f"No centroids detected in {imagePath}.")
                break
            imageAnalyzer = DetectionTool._imageAnalyzer
            imageAnalyzer._path = outputDir
            MetricTool = self.createMetricTool(imageAnalyzer)
            for _ in MetricTool.runPrefittingMetrics():
                pass            
            for bead in imageAnalyzer._beadAnalyzer:
                if bead._rejected == False and bead._roi is not None and bead._metricTool.meshBuilder is not None:
                    bead._metricTool.meshBuilder.saveMesh(os.path.join(self.getActivePath(bead._id, outputDir), f"bead_{bead._id}_mesh.obj"))
            FittingTool = self.createFittingTool(imageAnalyzer)
            FittingTool.computeFitting()
            for _ in MetricTool.runMetrics():
                pass
            FittingTool.displayFitting(outputDir)
            DetectionTool.cropPsf(outputDir)
            DetectionTool.GlobalCropPsf(outputDir)
            MetricTool.GenerateHeatmap(outputDir)
            self.generateReports(imageAnalyzer, outputDir)
            self._imageAnalzed.append(imageAnalyzer)
            print(f"Finished analyzing {imagePath}.")
        self.generatePDFReport()
        print("Batch analysis completed.")

    def drawParameterTableOnPDF(self, title, data, y):
        """Helper to draw a styled parameter table with a title
        
        Args:
            title (str): The section title
            data (List[List[str]]): The key-value pairs for the table
            y (int): The y-coordinate to start drawing (top to bottom)
        
        Returns:
            int: The new y-coordinate after drawing the table
        """
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, y, title)
        t = Table(data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        w, h = t.wrapOn(self.pdf, 0, 0)
        t.drawOn(self.pdf, 100, y - h - 10)
        return y - h - 50
    
    def generatePDFReport(self):
        """Generates a simplified PDF report for the batch analysis results, focusing on key metrics and bead analysis.
        
        Args:
            pdf (PDFReport, optional): An instance of the PDFReport class to generate the report. Defaults to None.
        """
        pdfPath = os.path.join(self._folderPath, f"Batch_analysis_results.pdf")
        self.pdf = canvas.Canvas(pdfPath, pagesize=A4)
        self.pdf.setTitle("Batch Analysis Results")
        self.pdf.setFont("Helvetica-Bold", 28)
        self.pdf.drawCentredString(300, 770, "Batch Analysis Report")
        self.pdf.setFont("Helvetica", 10)
        self.pdf.drawCentredString(300, 750, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        current_y = 700
        microscope_data = [
            ["Microscope type", f"{self._microscopeDatas.get('microscopeType', 'N/A')}"],
            ["Emission wavelength", f"{self._microscopeDatas.get('emissionWavelength', 'N/A')} nm"],
            ["Excitation wavelength", f"{self._microscopeDatas.get('excitationWavelength', 'N/A')} nm"],
            ["Refractive index", f"{self._microscopeDatas.get('refractiveIndex', 'N/A')}"],
            ["Numerical aperture", f"{self._microscopeDatas.get('numericalAperture', 'N/A')}"],
        ]
        current_y = self.drawParameterTableOnPDF("Microscope parameters", microscope_data, current_y)
        detection_data = [
            ["Detection method", f"{self._detectionDatas.get('detectionTool', 'N/A')}"],
            ["Minimal distance", f"{self._detectionDatas.get('minDist', 'N/A')}"],
            ["Sigma", f"{self._detectionDatas.get('sigma', 'N/A')}"],
            ["Threshold tool", f"{self._thresholdDatas.get('thresholdTool', 'N/A')}"],
            ["Threshold relative", f"{self._thresholdDatas.get('thresholdRel', 'N/A')}"],
        ]
        if self._thresholdDatas.get('thresholdTool') == 'manual':
            detection_data.append(["Threshold relative", f"{self._thresholdDatas.get('thresholdRel', 'N/A')}"])
        
        current_y = self.drawParameterTableOnPDF("Detection parameters", detection_data, current_y)
        extraction_data = [
            ["Bead size", f"{self._roiDatas.get('beadSize', 'N/A')}"],
            ["Crop factor", f"{self._roiDatas.get('cropFactor', 'N/A')}"],
            ["Distance ring-bead", f"{self._roiDatas.get('ringInnerDistance', 'N/A')}"],
            ["Ring thickness", f"{self._roiDatas.get('ringThickness', 'N/A')}"],
            ["Rejection distance", f"{self._roiDatas.get('rejectionDistance', 'N/A')}"],
            ["Intensity threshold", f"{self._roiDatas.get('thresholdIntensity', 'N/A')}"],
        ]
        current_y = self.drawParameterTableOnPDF("Extraction parameters", extraction_data, current_y)
        if current_y < 150:
            self.pdf.showPage()
            current_y = 750
        fitting_data = [
            ["Fitting type", f"{self._fittingDatas.get('fitType', 'N/A')}"],
            ["R² threshold", f"{self._fittingDatas.get('thresholdRSquared', 'N/A')}"],
            ["Prominence relative", f"{self._fittingDatas.get('prominenceRel', 'N/A')}"],
        ]
        self.drawParameterTableOnPDF("Fitting parameters", fitting_data, current_y)
        self.pdf.showPage()
        self.pdf.setFont("Helvetica-Bold", 28)
        self.pdf.drawCentredString(300, 770, "Batch Metrics Report")
        self.pdf.setFont("Helvetica", 10)
        current_y = 700
        meanSBR = sum(imageAnalyzer._meanSBR for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanContrast = sum(imageAnalyzer._meanContrast for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanEllipsRatio = sum(imageAnalyzer._meanEllipsRatio for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanOrientation = sum(imageAnalyzer._meanOrientation for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanLAR = sum(imageAnalyzer._meanLAR for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanSphericity = sum(imageAnalyzer._meanSphericity for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanComaticity = sum(imageAnalyzer._meanComaticity for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanSkeleton2Extremities = sum(imageAnalyzer._meanSkeleton2Extremities for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanRMin = sum(imageAnalyzer._meanRMin for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanAstigmatism = sum(imageAnalyzer._meanAstigmatism for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        meanSphericalAberration = sum(imageAnalyzer._meanSphericalAberration for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed)
        standardMetrics = [
            ["Mean SBR", f"{meanSBR:.2f}"],
            ["Mean Contrast", f"{meanContrast:.2f}"],
            ["Mean Ellips Ratio", f"{meanEllipsRatio:.2f}"],
            ["Mean Orientation", f"{meanOrientation:.2f}°"],
            ["Mean Lateral Asymmetry Ratio", f"{meanLAR:.2f}"],
            ["Mean Sphericity", f"{meanSphericity:.2f}"]
        ]
        current_y = self.drawParameterTableOnPDF("Standard metrics", standardMetrics, current_y)
        if current_y < 300:
            self.pdf.showPage()
            current_y = 700
        ComaticMetrics = [
            ["Mean Comaticity", f"{meanComaticity:.2f}"],
            ["Mean Skeleton to Extremities", f"{meanSkeleton2Extremities:.2f}"],
            ["Mean RMin", f"{meanRMin:.2f} µm"]
        ]
        current_y = self.drawParameterTableOnPDF("Comaticity metrics", ComaticMetrics, current_y)
        if current_y < 300:
            self.pdf.showPage()
            current_y = 700
        AstitmatismMetrics = [
            ["Mean Astigmatism", f"{meanAstigmatism:.2f}"],
        ]
        current_y = self.drawParameterTableOnPDF("Astigmatism metrics", AstitmatismMetrics, current_y)
        SphericalMetrics = [
            ["Mean Spherical Aberration", f"{meanSphericalAberration:.2f}"],
        ]
        current_y = self.drawParameterTableOnPDF("Spherical aberration metrics", SphericalMetrics, current_y)
        meanFWHM = [sum(imageAnalyzer._meanFWHM[i] for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed) for i in range(3)]
        meanDetermination = [sum(imageAnalyzer._meanDetermination[i] for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed) for i in range(3)]
        meanUncertainty = [sum(imageAnalyzer._meanUncertainty[i] for imageAnalyzer in self._imageAnalzed if imageAnalyzer is not None)/len(self._imageAnalzed) for i in range(3)]
        FittingMetrics = [
            ["Axis order", f"{', '.join(['Z', 'Y', 'X'])}"],
            ["Mean Determination (R²)", f"{', '.join(f'{x:.4f}' for x in meanDetermination)}"],
            ["Mean FWHM", f"{', '.join(f'{x:.4f}' for x in meanFWHM)}"],
            ["Mean Uncertainty", f"{', '.join(f'{(x[3] if isinstance(x, (list, tuple)) else 0.0):.4f}' for x in meanUncertainty)}"],
            ["Theoretical resolution", f"{', '.join(f'{x:.4f}' for x in self._imageAnalzed[0]._theoreticalResolution)} µm"],
            ["Sampling distance", f"{', '.join(f'{x:.4f}' for x in self._imageAnalzed[0]._samplingDistance)} µm"],
        ]
        current_y = self.drawParameterTableOnPDF("Fitting metrics", FittingMetrics, current_y)
        self.pdf.showPage()
        for imageAnalyzer in self._imageAnalzed:
            if imageAnalyzer is not None :
                imageAnalyzer.generateSimplifiedPDFReport(self.pdf)
                self.pdf.showPage()
        self.pdf.save()
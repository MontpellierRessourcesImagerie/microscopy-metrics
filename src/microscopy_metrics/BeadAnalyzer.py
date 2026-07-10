import os
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

from microscopy_metrics.fittingTools.fittingTool import FittingTool
from microscopy_metrics.metricTool.metricTool import MetricTool


class BeadAnalyzer(object):
    """Class to manage the analysis of individual beads, including storing the bead's image, region of interest (ROI), centroid, and results of fitting and metric calculations.
    It provides methods for running fitting and calculating metrics for the bead, which are used in the overall analysis of microscopy images.
    
    Attributes:
        _id (int): The unique identifier for the bead.
        _image (np.ndarray): The image data for the bead.
        _roi (list): The region of interest (ROI) for the bead in the image.
        _centroid (list): The centroid coordinates of the bead in the image.
        _rejected (bool): A flag indicating whether the bead has been rejected from analysis.
        _rejectionDesc (str): A description of why the bead was rejected, if applicable.
        _metricTool (MetricTool): An instance of the MetricTool class used for calculating metrics for the bead.
        _fitTool (FittingTool): An instance of the FittingTool class used for fitting curves to the bead's image data.
    """

    def __init__(self, id=0, image=None, roi=None, centroid=None):
        self._id = id
        self._image = image
        self._roi = roi
        self._centroid = centroid
        self._rejected = False
        self._rejectionDesc = ""
        self._metricTool = None
        self._fitTool = None

    def toDict(self):
        """Converts the bead analysis results into a dictionary format for easier access and manipulation of the bead's data, including its ID, image, ROI, centroid, rejection status, and any rejection descriptions.
        
        Returns:
            dict: A dictionary containing the bead's analysis results.
        """
        return {
            "id": self._id,
            "image": self._image,
            "roi": self._roi,
            "centroid": self._centroid,
            "rejected": self._rejected,
            "rejectionDesc": self._rejectionDesc,
        }

    def runFitting(
        self, fittingType="1D", spacing=[1, 1, 1], outputDir=None, prominenceRel=None
    ):
        """Runs the fitting process for the bead using the specified fitting type, spacing, output directory, and prominence relative value.
        
        Args:
            fittingType (str, optional): The type of fitting to perform. Defaults to "1D".
            spacing (list, optional): The spacing of the image pixels. Defaults to [1,1,1].
            outputDir (_type_, optional): The directory to save the fitting results. Defaults to None.
            prominenceRel (_type_, optional): The relative prominence value for the fitting. Defaults to None.
        """
        self._fitTool = FittingTool.getInstance(fittingType)
        self._fitTool._image = self._image
        self._fitTool._centroid = self._centroid
        self._fitTool._roi = self._roi
        self._fitTool._spacing = spacing
        self._fitTool._outputDir = outputDir
        if hasattr(self._fitTool, "_prominenceRel") and prominenceRel is not None:
            self._fitTool._prominenceRel = prominenceRel
        self._fitTool.processSingleFit(self._id)

    def runSBRMetric(self, spacing=[1, 1, 1], ringInnerDistance=1.0, ringThickness=2.0):
        """Runs the signal-to-background ratio (SBR) metric calculation for the bead using the specified spacing, ring inner distance, and ring thickness.
        
        Args:
            spacing (list, optional): The spacing of the image pixels. Defaults to [1,1,1].
            ringInnerDistance (float, optional): The inner distance of the ring for SBR calculation. Defaults to 1.0.
            ringThickness (float, optional): The thickness of the ring for SBR calculation. Defaults to 2.0.
        """
        self._metricTool = MetricTool()
        self._metricTool._image = self._image
        self._metricTool._pixelSize = spacing
        self._metricTool._ringInnerDistance = ringInnerDistance
        self._metricTool._ringThickness = ringThickness
        self._metricTool.processSingleSBRRing()

    def drawParameterTableOnPDF(self, pdf, title, data, y):
        """Helper to draw a styled parameter table with a title
        
        Args:
            pdf (reportlab.pdfgen.canvas.Canvas): The PDF canvas to draw on
            title (str): The section title
            data (List[List[str]]): The data rows
            y (int): The y-coordinate to start drawing (top to bottom)
        
        Returns:
            int: The new y-coordinate after drawing the table
        """
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawCentredString(300, y, title)
        y -= 10
    
        num_cols = len(data[0]) if data else 1
        if num_cols == 2:
            col_widths = [200, 200]
            header_col = True
            header_row = False
        elif num_cols == 4:
            col_widths = [160, 80, 80, 80]
            header_col = True
            header_row = True
        else:
            col_widths = [400 / max(num_cols, 1)] * num_cols
            header_col = False
            header_row = False
        style_cmds = [
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        if header_col and num_cols == 2:
            style_cmds.extend([
                ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ])
        elif header_row and num_cols == 4:
             style_cmds.extend([
                ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),   
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
             ])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle(style_cmds))
        w, h = t.wrapOn(pdf, 0, 0)
        t.drawOn(pdf, 100, y - h)
        return y - h - 35
    
    def generatePDFReport(self, pdf, inputDir, theoreticalResolution, samplingDistance):
        """Generates a clean PDF report for the bead analysis results.
        
        Args:
            pdf (reportlab.pdfgen.canvas.Canvas): The PDF canvas to draw on
            inputDir (str): The directory containing the input image files
            theoreticalResolution (float): The theoretical resolution of the microscope
            samplingDistance (float): The sampling distance of the microscope
        """
        beadPath = os.path.join(inputDir, f"bead_{self._id}")
        current_y = 800
        pdf.setFont("Helvetica-Bold", 24)
        pdf.drawCentredString(300, current_y, f"Bead Analysis Report - ID: {self._id}")
        current_y -= 40
        centroid_str = ', '.join(f'{c:.2f}' for c in self._centroid) if self._centroid is not None else 'Unknown'
        pdf.setFont("Helvetica", 12)
        pdf.drawCentredString(300, current_y, f"Centroid (Z, Y, X) : {centroid_str}")
        current_y -= 25
        if os.path.exists(os.path.join(beadPath, "Localisation.png")):
            img_w, img_h = 160, 160
            pdf.drawImage(os.path.join(beadPath, "Localisation.png"), 300 - (img_w/2), current_y - img_h, width=img_w, height=img_h, preserveAspectRatio=True)
            current_y -= (img_h + 35)
        metrics_data = [
            ["SBR", f"{self._metricTool._SBR:.2f}"],
            ["Contrast", f"{self._fitTool.contrast:.2f}"],
            ["Ellipticity ratio", f"{self._metricTool._ellipsRatio:.2f}"],
            ["Sphericity", f"{self._metricTool._sphericity:.2f}"],
            ["Lateral asymmetry", f"{self._metricTool._LAR:.2f}"],
            ["Orientation", f"{self._metricTool._orientation:.2f}°"],
            ["Comaticity", f"{self._metricTool._comaticity:.2f}"],
            ["Skeleton/Extremities ratio", f"{self._metricTool._skeleton2Extremities:.2f}"],
            ["Concavity", f"{self._metricTool.meshBuilder._concavity:.2f}" if hasattr(self._metricTool, 'meshBuilder') and self._metricTool.meshBuilder is not None and self._metricTool.meshBuilder._concavity is not None else "N/A"],
            ["Astigmatism", f"{self._metricTool._astigmatism:.2f}"],
            ["Spherical aberration", f"{self._metricTool._sphericalAberration:.2f}"],
        ]
        rmin_val = "Inf" if getattr(self._metricTool, '_RMin', 0) == float('inf') else f"{getattr(self._metricTool, '_RMin', 0):.2f}"
        metrics_data.insert(8, ["Minimal Curvature Radius (RMin)", f"{rmin_val} µm"])
        current_y = self.drawParameterTableOnPDF(pdf, "Quality Metrics", metrics_data, current_y)
        if self._metricTool._commentary:
            pdf.setFont("Helvetica", 10)
            commentary_lines = self._metricTool._commentary.strip().split('\n')
            for line in commentary_lines:
                if current_y < 50:
                    pdf.showPage()
                    current_y = 800
                pdf.drawString(100, current_y, line)
                current_y -= 12
        current_y -= 20
        fittingData = [
            ["Axis", "Z", "Y", "X"],
            ["Theoretical res. (µm)", f"{theoreticalResolution[0]:.4f}", f"{theoreticalResolution[1]:.4f}", f"{theoreticalResolution[2]:.4f}"],
            ["Sampling dist. (µm)",   f"{samplingDistance[0]:.4f}",      f"{samplingDistance[1]:.4f}",      f"{samplingDistance[2]:.4f}"],
            ["FWHM (µm)",             f"{self._fitTool.fwhms[0]:.4f}",   f"{self._fitTool.fwhms[1]:.4f}",   f"{self._fitTool.fwhms[2]:.4f}"],
            ["Uncertainty",           f"{self._fitTool.uncertainties[0][3]:.4f}", f"{self._fitTool.uncertainties[1][3]:.4f}", f"{self._fitTool.uncertainties[2][3]:.4f}"],
            ["Determination (R²)",    f"{self._fitTool.determinations[0]:.4f}", f"{self._fitTool.determinations[1]:.4f}", f"{self._fitTool.determinations[2]:.4f}"],
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Gaussian Fitting Results", fittingData, current_y)
        if self._fitTool._commentary:
            current_y -= 15
            pdf.setFont("Helvetica", 10)
            commentary_lines = self._fitTool._commentary.strip().split('\n')
            for line in commentary_lines:
                if current_y < 50:
                    pdf.showPage()
                    current_y = 800
                pdf.drawString(100, current_y, line)
                current_y -= 12
        pdf.showPage()
        current_y = 800
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawCentredString(300, current_y, "Intensity Profiles and Projections")
        current_y -= 40
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(300, current_y, "Center Projections (XY, XZ, YZ Planes)")
        current_y -= 15
        img_size = 170
        x_positions = [25, 212, 400] 
        views = [
            {"file": "XY_view.png", "title": "XY View"},
            {"file": "XZ_view.png", "title": "XZ View"},
            {"file": "YZ_view.png", "title": "YZ View"},
        ]
        current_y -= img_size
        y_images = current_y
        for i, view in enumerate(views):
            x = x_positions[i]
            if os.path.exists(os.path.join(beadPath, view["file"])):
                pdf.drawImage(os.path.join(beadPath, view["file"]), x, y_images, width=img_size, height=img_size, preserveAspectRatio=True)
            pdf.setFont("Helvetica", 10)
            pdf.drawCentredString(x + (img_size / 2), y_images - 15, view["title"])
        current_y -= 60
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(300, current_y, "Gaussian Fits (Z, Y, X Axes)")
        current_y -= 15
        current_y -= img_size
        y_fits = current_y
        fits = [
            {"file": "fit_curve_1D_Z.png", "title": "Z-Axis Fit"},
            {"file": "fit_curve_1D_Y.png", "title": "Y-Axis Fit"},
            {"file": "fit_curve_1D_X.png", "title": "X-Axis Fit"},
        ]
        for i, fit in enumerate(fits):
            x = x_positions[i]
            if os.path.exists(os.path.join(beadPath, fit["file"])):
                pdf.drawImage(os.path.join(beadPath, fit["file"]), x, y_fits, width=img_size, height=img_size, preserveAspectRatio=True)
            pdf.setFont("Helvetica", 10)
            pdf.drawCentredString(x + (img_size / 2), y_fits - 15, fit["title"])
        
        current_y -= 60
        
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(300, current_y, "Tilted Gaussian Fits (XZ, ZY Planes)")
        current_y -= 30
        current_y -= img_size * 1.3
        img_size = img_size * 1.2
        y_tilted_fits = current_y
        tilted_fits = [
            {"file": "2D_Gaussian_Image_XZ.png", "title": "Tilted XZ Fit"},
            {"file": "2D_Gaussian_Image_ZY.png", "title": "Tilted ZY Fit"},
        ]
        x_positions_tilted = [75, 350]
        for i, fit in enumerate(tilted_fits):
            x = x_positions_tilted[i]
            if os.path.exists(os.path.join(beadPath, fit["file"])):
                pdf.drawImage(os.path.join(beadPath, fit["file"]), x, y_tilted_fits, width=img_size, height=img_size, preserveAspectRatio=True)
                pdf.setFont("Helvetica", 10)
                pdf.drawCentredString(x + (img_size / 2), y_tilted_fits - 15, fit["title"])
            else :
                pdf.setFont("Helvetica", 10)
                pdf.drawCentredString(x + (img_size / 2), y_tilted_fits - 15, f"{fit['title']} (Not Available)")
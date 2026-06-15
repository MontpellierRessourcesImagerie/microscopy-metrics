import os
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle

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
        self._samplingDistance = [0.0, 0.0, 0.0]
        self._density = 0.0
        
        self._meanSBR = 0.0
        self._meanContrast = 0.0
        self._meanEllipsRatio = 0.0
        self._meanOrientation = 0.0
        self._meanLAR = 0.0
        self._meanSphericity = 0.0
        
        self._meanComaticity = 0.0
        self._meanSkeleton2Extremities = 0.0
        self._meanRMin = 0.0
        self._meanConcavity = 0.0
        
        self._meanAstigmatism = 0.0
        
        self._meanSphericalAberration = 0.0
        
        self._meanDetermination = [0.0, 0.0, 0.0]
        self._meanFWHM = [0.0, 0.0, 0.0]
        self._meanUncertainty = [0.0, 0.0, 0.0]

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
            "meanSkeleton2Extremities": self._meanSkeleton2Extremities,
            "meanRMin": self._meanRMin,
            "meanDetermination": self._meanDetermination,
            "meanFWHM": self._meanFWHM,
            "meanUncertainty": self._meanUncertainty,
            "meanLAR": self._meanLAR,
            "meanSphericity": self._meanSphericity,
        }
    
    def drawParameterTableOnPDF(self,pdf, title, data, y):
        """Helper to draw a styled parameter table with a title
        Args:
            title (str): The section title
            data (List[List[str]]): The key-value pairs for the table
            y (int): The y-coordinate to start drawing (top to bottom)
        Returns:
            int: The new y-coordinate after drawing the table
        """
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawCentredString(300, y, title)
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
        w, h = t.wrapOn(pdf, 0, 0)
        t.drawOn(pdf, 100, y - h - 10)
        return y - h - 50
    
    def generatePDFReport(self, pdf):
        """Generates a PDF report for the image analysis results, including bead analysis and various metrics.
        Args:
            pdf (PDFReport, optional): An instance of the PDFReport class to generate the report. Defaults to None.
        """
        pdf.setFont("Helvetica-Bold", 28)
        pdf.drawCentredString(300, 770, "Analysis Parameters Report")
        pdf.setFont("Helvetica", 10)
        current_y = 700
        if os.path.exists(os.path.join(self._path, "Localisation.png")):
            pdf.drawImage(
                os.path.join(self._path, "Localisation.png"),
                172,
                current_y - 300,
                width=300,
                height=300,
                preserveAspectRatio=True,
            )
        current_y -= 350
        styles = getSampleStyleSheet()
        styleN = styles["Normal"]
        styleN.wordWrap = 'CJK' 
        imgData = [
            ["Image path", Paragraph(f"{self._path}", styleN)],
            ["Image shape", f"{self._image.shape if self._image is not None else 'N/A'}"],
            ["Bead size", f"{self._beadSize} µm"],
            ["Pixel size", f"{', '.join(f'{x:.4f}' for x in self._pixelSize)} µm"],
            ["Number of beads analyzed", f"{len([bead for bead in self._beadAnalyzer if not bead._rejected])}"],
            ["Number of beads rejected", f"{len([bead for bead in self._beadAnalyzer if bead._rejected])}"]
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Image parameters", imgData, current_y)
        if current_y < 300:
            pdf.showPage()
            current_y = 700
        standardMetrics = [
            ["Mean SBR", f"{self._meanSBR:.2f}"],
            ["Mean Contrast", f"{self._meanContrast:.2f}"],
            ["Mean Ellips Ratio", f"{self._meanEllipsRatio:.2f}"],
            ["Mean Orientation", f"{self._meanOrientation:.2f}°"],
            ["Mean Lateral Asymmetry Ratio", f"{self._meanLAR:.2f}"],
            ["Mean Sphericity", f"{self._meanSphericity:.2f}"]
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Standard metrics", standardMetrics, current_y)
        if current_y < 300:
            pdf.showPage()
            current_y = 700
        ComaticMetrics = [
            ["Mean Comaticity", f"{self._meanComaticity:.2f}"],
            ["Mean Skeleton to Extremities", f"{self._meanSkeleton2Extremities:.2f}"],
            ["Mean RMin", f"{self._meanRMin:.2f} µm"]
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Comaticity metrics", ComaticMetrics, current_y)
        if current_y < 300:
            pdf.showPage()
            current_y = 700
        AstitmatismMetrics = [
            ["Mean Astigmatism", f"{self._meanAstigmatism:.2f}"],
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Astigmatism metrics", AstitmatismMetrics, current_y)
        SphericalMetrics = [
            ["Mean Spherical Aberration", f"{self._meanSphericalAberration:.2f}"],
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Spherical aberration metrics", SphericalMetrics, current_y)
        FittingMetrics = [
            ["Axis order", f"{', '.join(['Z', 'Y', 'X'])}"],
            ["Mean Determination (R²)", f"{', '.join(f'{x:.4f}' for x in self._meanDetermination)}"],
            ["Mean FWHM", f"{', '.join(f'{x:.4f}' for x in self._meanFWHM)}"],
            ["Mean Uncertainty", f"{', '.join(f'{x[3]:.4f}' for x in self._meanUncertainty)}"],
            ["Theoretical resolution", f"{', '.join(f'{x:.4f}' for x in self._theoreticalResolution)} µm"],
            ["Sampling distance", f"{', '.join(f'{x:.4f}' for x in self._samplingDistance)} µm"],
        ]
        current_y = self.drawParameterTableOnPDF(pdf, "Fitting metrics", FittingMetrics, current_y)
        pdf.showPage()
        pdf.setFont("Helvetica-Bold", 28)
        pdf.drawCentredString(300, 770, "Metrics Heatmap")
        heatmap_files = [
            "SBR_Heatmap.png",
            "Contrast_Heatmap.png",
            "EllipsRatio_Heatmap.png",
            "LAR_Heatmap.png",
            "Sphericity_Heatmap.png",
            "Orientation_Heatmap.png",
            "Comaticity_Heatmap.png",
            "Skeleton2Extremities_Heatmap.png",
            "RMin_Heatmap.png",
            "Astigmatism_Heatmap.png",
            "SphericalAberration_Heatmap.png",
        ]
        existing_heatmaps = [f for f in heatmap_files if os.path.exists(os.path.join(self._path, f))]
        img_width = 250
        img_height = 250
        x_positions = [40, 310]
        y_start = 480
        y_step = 280

        for i, file_name in enumerate(existing_heatmaps):
            if i > 0 and i % 4 == 0:
                pdf.showPage()
            col_index = i % 2 
            row_index = (i % 4) // 2  
            x = x_positions[col_index]
            y = y_start - (row_index * y_step)
            pdf.drawImage(
                os.path.join(self._path, file_name),
                x,
                y,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True,
            )
        pdf.showPage()
        
        rejectedBeads = [bead for bead in self._beadAnalyzer if bead._rejected]
        if rejectedBeads:
            MAX_ROWS_PER_PAGE = 30
            
            for i in range(0, len(rejectedBeads), MAX_ROWS_PER_PAGE):
                chunk = rejectedBeads[i : i + MAX_ROWS_PER_PAGE]
                title = "Rejected beads" if i == 0 else "Rejected beads"
                rejectedData = [
                    ["Bead ID", "Rejection reason"],
                    *[[bead._id, bead._rejectionDesc] for bead in chunk]
                ]
                self.drawParameterTableOnPDF(pdf, title, rejectedData, 770)
                if i + MAX_ROWS_PER_PAGE < len(rejectedBeads):
                    pdf.showPage()
            pdf.showPage()
        for bead in self._beadAnalyzer:
            if not bead._rejected:
                bead.generatePDFReport(pdf, self._path, self._theoreticalResolution, self._samplingDistance)
                pdf.showPage()
        
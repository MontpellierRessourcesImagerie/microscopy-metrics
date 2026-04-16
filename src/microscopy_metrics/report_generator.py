from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle
import os
from jinja2 import Environment, FileSystemLoader
import csv
import numpy as np


class ReportGenerator(object):
    def __init__(self):
        self._outputDir = ""
        self._outputPath = ""
        self._analysisData = {}
        self._parametersDetection = {}
        self._parametersAcquisition = {}
        self._filteredBeads = []
        self._meanSBR = 0.0
        self._theoreticalResolution = []
        self._imageShape = None

        self.pdf = None

    @property
    def outputDir(self):
        return self._outputDir

    @outputDir.setter
    def outputDir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The outputDir is wrong")
        self._outputDir = value

    @property
    def outputPath(self):
        return self._outputPath

    @outputPath.setter
    def outputPath(self, value):
        if value is None:
            raise ValueError("The outputDir is wrong")
        self._outputPath = value

    @property
    def analysisData(self):
        return self._analysisData

    @analysisData.setter
    def analysisData(self, data):
        if data is None or data == {}:
            raise ValueError("analysisData must be a not empty collection")
        self._analysisData = data

    @property
    def parametersDetection(self):
        return self._parametersDetection

    @parametersDetection.setter
    def parametersDetection(self, data):
        if data is None or data == {}:
            raise ValueError("parametersDetection must be a not empty collection")
        self._parametersDetection = data

    @property
    def parametersAcquisition(self):
        return self._parametersAcquisition

    @parametersAcquisition.setter
    def parametersAcquisition(self, data):
        if data is None or data == {}:
            raise ValueError("parametersAcquisition must be a not empty collection")
        self._parametersAcquisition = data

    @property
    def filteredBeads(self):
        return self._filteredBeads

    @filteredBeads.setter
    def filteredBeads(self, data):
        if len(data) == 0:
            raise ValueError("filteredBeads must be a not empty list")
        self._filteredBeads = data

    @property
    def meanSBR(self):
        return self._meanSBR

    @meanSBR.setter
    def meanSBR(self, value):
        if not isinstance(value, float):
            raise ValueError("Please enter a correct value for meanSBR")
        self._meanSBR = value

    @property
    def theoreticalResolution(self):
        return self._theoreticalResolution

    @theoreticalResolution.setter
    def theoreticalResolution(self, data):
        if not isinstance(data, list):
            raise ValueError("Invalid format for theoretical resolution")
        self._theoreticalResolution = data

    def getActivePath(self, index):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        activePath = os.path.join(self._outputDir, f"bead_{index}")
        if not os.path.exists(activePath):
            os.makedirs(activePath)
        return activePath

    def drawParagaphOnPDF(self, textLines, x, y):
        """Helper to quickly write a paragraph on the pdf

        Args:
            textLines (List(String)): The list of lines we want to write on the pdf
            x (int): x coordinate of the paragraph
            y (int): y coordinate of the paragraph
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        fullText = "<br/>".join(textLines)
        p = Paragraph(fullText, normalStyle)
        p.wrapOn(self.pdf, 500, 100)
        p.drawOn(self.pdf, x, y)

    def drawTableOnPDF(self, data):
        """Helper to quickly add a table on the pdf

        Args:
            data (Matrix(String)): The table to write on the pdf
        """
        s = getSampleStyleSheet()
        s = s["BodyText"]
        s.wordWrap = "CJK"
        data2 = [[Paragraph(cell, s) for cell in row] for row in data]
        t = Table(data=data2, colWidths=[80, 50, 50, 50])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        t.wrapOn(self.pdf, 0, 0)
        t.drawOn(self.pdf, 40, 600)

    def drawSingleBeadReportPDF(self, activePath, index):
        """Function to generate the report of a bead and add it to the pdf

        Args:
            activePath (Path): Path to the folder of the current image
            index (int): Bead ID corresping to it's position in the list
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(150, 770, f"Bead {index}")
        self.pdf.drawImage(
            os.path.join(activePath, "Localisation.png"),
            350,
            600,
            width=200,
            height=200,
            preserveAspectRatio=True,
        )
        textLines = [
            f"centroid: {self._filteredBeads[self._analysisData[index]['id']]}",
            f"Signal to Background ratio: {self._analysisData[index]['SBR']:.4f}",
            f"Lateral asymmetry ratio: {self._analysisData[index]['LAR']:.4f}",
            f"Sphericity: {self._analysisData[index]['sphericity']:.4f}",
        ]
        text = self.pdf.beginText(40, 750)
        text.setFont(normalStyle.fontName, normalStyle.fontSize)
        for line in textLines:
            text.textLine(line)
        self.pdf.drawText(text)
        data = [
            ["", "Z", "Y", "X"],
            [
                "Theoretical resolution",
                f"{self._theoreticalResolution[0]:.4f}",
                f"{self._theoreticalResolution[1]:.4f}",
                f"{self._theoreticalResolution[2]:.4f}",
            ],
            [
                "FWHM (µm)",
                f"{self._analysisData[index]['FWHM'][0]:.4f}",
                f"{self._analysisData[index]['FWHM'][1]:.4f}",
                f"{self._analysisData[index]['FWHM'][2]:.4f}",
            ],
            [
                "Uncertainty",
                f"{self._analysisData[index]['uncertainty'][0][3]:.4f}",
                f"{self._analysisData[index]['uncertainty'][1][3]:.4f}",
                f"{self._analysisData[index]['uncertainty'][2][3]:.4f}",
            ],
            [
                "Determination",
                f"{self._analysisData[index]['determination'][0]:.4f}",
                f"{self._analysisData[index]['determination'][1]:.4f}",
                f"{self._analysisData[index]['determination'][2]:.4f}",
            ],
        ]
        self.drawTableOnPDF(data)
        self.pdf.drawImage(
            os.path.join(activePath, "YZ_view.png"),
            50,
            425,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(activePath, "fit_curve_1D_X.png"),
            300,
            375,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(activePath, "XZ_view.png"),
            50,
            225,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(activePath, "fit_curve_1D_Y.png"),
            300,
            175,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(activePath, "XY_view.png"),
            50,
            25,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(activePath, "fit_curve_1D_Z.png"),
            300,
            -25,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.showPage()

    def generatePDFReport(self, imagePath):
        """
        Args:
            imagePath (Path): Absolute path of the image
        """
        rois = [entry["ROI"] for entry in self._analysisData]
        self.pdf = canvas.Canvas(self._outputPath, pagesize=A4)
        self.pdf.setTitle("PSF analysis results")
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(300, 770, "Analysis Results")
        textLines = [
            f"Image location: {imagePath}",
            f"Identified beads: {len(self._filteredBeads)}",
            f"Extracted ROIs: {len(rois)}",
            f"Signal to background ratio: {self._meanSBR:.2f}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 680)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 600, "Acquisition parameters")
        textLines = [
            f"Pixel size: [{self._parametersAcquisition['Pixel size Z']},{self._parametersAcquisition['Pixel size Y']},{self._parametersAcquisition['Pixel size X']}]",
            f"Image shape: [{self._imageShape[0]},{self._imageShape[1]},{self._imageShape[2]}]",
            f"Microscope type: {self._parametersAcquisition['Microscope type']}",
            f"Emission wavelength: {self._parametersAcquisition['Emission wavelength']}nm",
            f"Refractive index: {self._parametersAcquisition['Refraction index']}",
            f"Numerical aperture: {self._parametersAcquisition['Numerical aperture']}",
        ]
        self.drawParagaphOnPDF(textLines, 40, 500)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 400, "Detection parameters")
        textLines = [
            f"Detection method: {self._parametersDetection['Detection tool']}"
        ]
        if self._parametersDetection['Detection tool'] == "peak local maxima":
            textLines.append(
                f"Minimal distance: {self._parametersDetection['Min dist']}"
            )
        else:
            textLines.append(f"Sigma: {self._parametersDetection['Sigma']}")
        textLines.extend(
            [
                f"Bead size: {self._parametersDetection['Theoretical bead size (µm)']}",
                f"Crop factor: {self._parametersDetection['crop factor']}",
            ]
        )
        if self._parametersDetection["Threshold"] != "manual":
            textLines.append(
                f"Threshold tool: {self._parametersDetection['Threshold']}"
            )
        else:
            textLines.append(
                f"Threshold relative: {self._parametersDetection['threshold']}"
            )
        textLines.extend(
            [
                f"Distance ring-bead: {self._parametersDetection['Inner annulus distance to bead (µm)']}",
                f"Ring thickness: {self._parametersDetection['Annulus thickness (µm)']}",
            ]
        )
        self.drawParagaphOnPDF(textLines, 40, 300)

        self.pdf.showPage()

        for i, _ in enumerate(rois):
            activePath = self.getActivePath(index=i)
            # Generating the HTML report
            self.drawSingleBeadReportPDF(activePath, i)

        self.pdf.save()

    def generateHTMLReport(self):
        for i, psf in enumerate(self._analysisData):
            path = self.getActivePath(i)
            activePath = os.path.join(path, "PSF_analysis_result.html")
            templateDir = os.path.join(os.path.dirname(__file__), "res", "template")
            env = Environment(loader=FileSystemLoader(templateDir))
            template = env.get_template("report_template.html")
            data = {
                "title": i,
                "bead": self._filteredBeads[psf["id"]],
                "results": psf,
                "path": path,
                "theoretical_resolution": self._theoreticalResolution,
            }
            htmlContent = template.render(data)
            with open(activePath, "w") as f:
                f.write(htmlContent)

    def generateCSVReport(self, outputPath):
        """
        Args:
            outputPath (Path): Path of the generated csv file
        """
        with open(outputPath, mode="w", newline="") as file:
            writer = csv.writer(file)
            for i, psf in enumerate(self._analysisData):
                writer.writerow([f"Bead {i}"])
                dataBead = [
                    ["", "Z", "Y", "X"],
                    [
                        "Centroid coordinates",
                        f"{self._filteredBeads[i][0]}",
                        f"{self._filteredBeads[i][1]}",
                        f"{self._filteredBeads[i][2]}",
                    ],
                ]
                dataFitting = [
                    ["", "Z", "Y", "X"],
                    [
                        "Theoretical resolution",
                        f"{self._theoreticalResolution[0]}",
                        f"{self._theoreticalResolution[1]}",
                        f"{self._theoreticalResolution[2]}",
                    ],
                    [
                        "FWHM",
                        f"{self._analysisData[i]['FWHM'][0]}",
                        f"{self._analysisData[i]['FWHM'][1]}",
                        f"{self._analysisData[i]['FWHM'][2]}",
                    ],
                    [
                        "Uncertainty",
                        f"{self._analysisData[i]['uncertainty'][0][3]}",
                        f"{self._analysisData[i]['uncertainty'][1][3]}",
                        f"{self._analysisData[i]['uncertainty'][2][3]}",
                    ],
                    [
                        "Determination",
                        f"{self._analysisData[i]['determination'][0]}",
                        f"{self._analysisData[i]['determination'][1]}",
                        f"{self._analysisData[i]['determination'][2]}",
                    ],
                ]
                dataMetrics = [
                    ["Metric", "Value"],
                    ["Signal to background ratio", f"{self._analysisData[i]['SBR']}"],
                    ["Lateral asymmetry ratio", f"{self._analysisData[i]['LAR']}"],
                    ["Sphericity", f"{self._analysisData[i]['sphericity']}"],
                ]
                writer.writerow(["Bead's datas"])
                writer.writerows(dataBead)
                writer.writerow([])
                writer.writerow(["Fitting's datas"])
                writer.writerows(dataFitting)
                writer.writerow([])
                writer.writerow(["Meric's results"])
                writer.writerows(dataMetrics)
                writer.writerow([])
                writer.writerow([])

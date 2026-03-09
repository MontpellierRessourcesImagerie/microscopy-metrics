from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle
import os
from jinja2 import Environment, FileSystemLoader
import csv
import numpy as np


class Report_Generator(object):
    def __init__(self):
        self._output_dir = ""
        self._output_path = ""
        self._analysis_data = {}
        self._parameters_detection = {}
        self._parameters_acquisition = {}
        self._filtered_beads = []
        self._mean_SBR = 0.0
        self._theoretical_resolution = []
        self.image_shape = None

        self.pdf = None

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None or not os.path.exists(value):
            raise ValueError("The output_dir is wrong")
        self._output_dir = value

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        if value is None:
            raise ValueError("The output_dir is wrong")
        self._output_path = value

    @property
    def analysis_data(self):
        return self._analysis_data

    @analysis_data.setter
    def analysis_data(self, data):
        if data is None or data == {}:
            raise ValueError("analysis_data must be a not empty collection")
        self._analysis_data = data

    @property
    def parameters_detection(self):
        return self._parameters_detection

    @parameters_detection.setter
    def parameters_detection(self, data):
        if data is None or data == {}:
            raise ValueError("parameters_detection must be a not empty collection")
        self._parameters_detection = data

    @property
    def parameters_acquisition(self):
        return self._parameters_acquisition

    @parameters_acquisition.setter
    def parameters_acquisition(self, data):
        if data is None or data == {}:
            raise ValueError("parameters_acquisition must be a not empty collection")
        self._parameters_acquisition = data

    @property
    def filtered_beads(self):
        return self._filtered_beads

    @filtered_beads.setter
    def filtered_beads(self, data):
        if not isinstance(data, np.ndarray) or len(data) == 0:
            raise ValueError("filtered_beads must be a not empty list")
        self._filtered_beads = data

    @property
    def mean_SBR(self):
        return self._mean_SBR

    @mean_SBR.setter
    def mean_SBR(self, value):
        if not isinstance(value, float):
            raise ValueError("Please enter a correct value for mean_SBR")
        self._mean_SBR = value

    @property
    def theoretical_resolution(self):
        return self._theoretical_resolution

    @theoretical_resolution.setter
    def theoretical_resolution(self, data):
        if not isinstance(data, list):
            raise ValueError("Invalid format for theoretical resolution")
        self._theoretical_resolution = data

    def get_active_path(self, index):
        """
        Args:
            index (int): Bead ID corresping to it's position in the list

        Returns:
            Path: Folder's path found (or created) for the selected bead
        """
        active_path = os.path.join(self._output_dir, f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path

    def draw_paragaph_on_pdf(self, textLines, x, y):
        """Helper to quickly write a paragraph on the pdf

        Args:
            textLines (List(String)): The list of lines we want to write on the pdf
            x (int): x coordinate of the paragraph
            y (int): y coordinate of the paragraph
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        full_text = "<br/>".join(textLines)
        p = Paragraph(full_text, normalStyle)
        p.wrapOn(self.pdf, 500, 100)
        p.drawOn(self.pdf, x, y)

    def draw_table_on_pdf(self, data):
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

    def draw_single_bead_report_pdf(self, active_path, index):
        """Function to generate the report of a bead and add it to the pdf

        Args:
            active_path (Path): Path to the folder of the current image
            index (int): Bead ID corresping to it's position in the list
        """
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet["Normal"]
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(150, 770, f"Bead {index}")
        self.pdf.drawImage(
            os.path.join(active_path, "Localisation.png"),
            350,
            600,
            width=200,
            height=200,
            preserveAspectRatio=True,
        )
        textLines = [
            f"centroid: {self._filtered_beads[self._analysis_data[index]["id"]]}",
            f"Signal to Background ratio: {self._analysis_data[index]["SBR"]:.4f}",
            f"Lateral asymmetry ratio: {self._analysis_data[index]["LAR"]:.4f}",
            f"Sphericity: {self._analysis_data[index]["sphericity"]:.4f}",
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
                f"{self._theoretical_resolution[0]:.4f}",
                f"{self._theoretical_resolution[1]:.4f}",
                f"{self._theoretical_resolution[2]:.4f}",
            ],
            [
                "FWHM (µm)",
                f"{self._analysis_data[index]["FWHM"][0]:.4f}",
                f"{self._analysis_data[index]["FWHM"][1]:.4f}",
                f"{self._analysis_data[index]["FWHM"][2]:.4f}",
            ],
            [
                "Uncertainty",
                f"{self._analysis_data[index]["uncertainty"][0][3]:.4f}",
                f"{self._analysis_data[index]["uncertainty"][1][3]:.4f}",
                f"{self._analysis_data[index]["uncertainty"][2][3]:.4f}",
            ],
            [
                "Determination",
                f"{self._analysis_data[index]["determination"][0]:.4f}",
                f"{self._analysis_data[index]["determination"][1]:.4f}",
                f"{self._analysis_data[index]["determination"][2]:.4f}",
            ],
        ]
        self.draw_table_on_pdf(data)
        self.pdf.drawImage(
            os.path.join(active_path, "YZ_view.png"),
            50,
            425,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(active_path, "fit_curve_1D_X.png"),
            300,
            375,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(active_path, "XZ_view.png"),
            50,
            225,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(active_path, "fit_curve_1D_Y.png"),
            300,
            175,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(active_path, "XY_view.png"),
            50,
            25,
            width=150,
            height=150,
            preserveAspectRatio=True,
        )
        self.pdf.drawImage(
            os.path.join(active_path, "fit_curve_1D_Z.png"),
            300,
            -25,
            width=250,
            height=250,
            preserveAspectRatio=True,
        )
        self.pdf.showPage()

    def generate_pdf_report(self, image_path):
        """
        Args:
            image_path (Path): Absolute path of the image
        """
        rois = [entry["ROI"] for entry in self._analysis_data]
        self.pdf = canvas.Canvas(self._output_path, pagesize=A4)
        self.pdf.setTitle("PSF analysis results")
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(300, 770, "Analysis Results")
        textLines = [
            f"Image location: {image_path}",
            f"Identified beads: {len(self._filtered_beads)}",
            f"Extracted ROIs: {len(rois)}",
            f"Signal to background ratio: {self._mean_SBR:.2f}",
        ]
        self.draw_paragaph_on_pdf(textLines, 40, 680)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 600, "Acquisition parameters")
        textLines = [
            f"Pixel size: [{self._parameters_acquisition["Pixel size Z"]},{self._parameters_acquisition["Pixel size Y"]},{self._parameters_acquisition["Pixel size X"]}]",
            f"Image shape: [{self.image_shape[0]},{self.image_shape[1]},{self.image_shape[2]}]",
            f"Microscope type: {self._parameters_acquisition["Microscope type"]}",
            f"Emission wavelength: {self._parameters_acquisition["Emission wavelength"]}nm",
            f"Refractive index: {self._parameters_acquisition["Refraction index"]}",
            f"Numerical aperture: {self._parameters_acquisition["Numerical aperture"]}",
        ]
        self.draw_paragaph_on_pdf(textLines, 40, 500)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300, 400, "Detection parameters")
        textLines = [
            f"Detection method: {self._parameters_detection['Detection tool']}"
        ]
        if self._parameters_detection['Detection tool'] == "peak local maxima":
            textLines.append(
                f"Minimal distance: {self._parameters_detection['Min dist']}"
            )
        else:
            textLines.append(f"Sigma: {self._parameters_detection['Sigma']}")
        textLines.extend(
            [
                f"Bead size: {self._parameters_detection['Theoretical bead size (µm)']}",
                f"Crop factor: {self._parameters_detection['crop factor']}",
            ]
        )
        if self._parameters_detection["Threshold"] != "manual":
            textLines.append(
                f"Threshold tool: {self._parameters_detection['Threshold']}"
            )
        else:
            textLines.append(
                f"Threshold relative: {self._parameters_detection['threshold']}"
            )
        textLines.extend(
            [
                f"Distance ring-bead: {self._parameters_detection['Inner annulus distance to bead (µm)']}",
                f"Ring thickness: {self._parameters_detection['Annulus thickness (µm)']}",
            ]
        )
        self.draw_paragaph_on_pdf(textLines, 40, 300)

        self.pdf.showPage()

        for i, _ in enumerate(rois):
            active_path = self.get_active_path(index=i)
            # Generating the HTML report
            self.draw_single_bead_report_pdf(active_path, i)

        self.pdf.save()

    def generate_html_report(self):
        for i, psf in enumerate(self._analysis_data):
            path = self.get_active_path(i)
            active_path = os.path.join(path, "PSF_analysis_result.html")
            template_dir = os.path.join(os.path.dirname(__file__), "res", "template")
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template("report_template.html")
            data = {
                "title": i,
                "bead": self._filtered_beads[psf["id"]],
                "results": psf,
                "path": path,
                "theoretical_resolution": self._theoretical_resolution,
            }
            html_content = template.render(data)
            with open(active_path, "w") as f:
                f.write(html_content)

    def generate_csv_report(self, output_path):
        """
        Args:
            output_path (Path): Path of the generated csv file
        """
        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            for i, psf in enumerate(self._analysis_data):
                writer.writerow([f"Bead {i}"])
                data_bead = [
                    ["", "Z", "Y", "X"],
                    [
                        "Centroid coordinates",
                        f"{self._filtered_beads[i][0]}",
                        f"{self._filtered_beads[i][1]}",
                        f"{self._filtered_beads[i][2]}",
                    ],
                ]
                data_fitting = [
                    ["", "Z", "Y", "X"],
                    [
                        "Theoretical resolution",
                        f"{self._theoretical_resolution[0]}",
                        f"{self._theoretical_resolution[1]}",
                        f"{self._theoretical_resolution[2]}",
                    ],
                    [
                        "FWHM",
                        f"{self._analysis_data[i]["FWHM"][0]}",
                        f"{self._analysis_data[i]["FWHM"][1]}",
                        f"{self._analysis_data[i]["FWHM"][2]}",
                    ],
                    [
                        "Uncertainty",
                        f"{self._analysis_data[i]["uncertainty"][0][3]}",
                        f"{self._analysis_data[i]["uncertainty"][1][3]}",
                        f"{self._analysis_data[i]["uncertainty"][2][3]}",
                    ],
                    [
                        "Determination",
                        f"{self._analysis_data[i]["determination"][0]}",
                        f"{self._analysis_data[i]["determination"][1]}",
                        f"{self._analysis_data[i]["determination"][2]}",
                    ],
                ]
                data_metrics = [
                    ["Metric", "Value"],
                    ["Signal to background ratio", f"{self._analysis_data[i]["SBR"]}"],
                    ["Lateral asymmetry ratio", f"{self._analysis_data[i]["LAR"]}"],
                    ["Sphericity", f"{self._analysis_data[i]["sphericity"]}"],
                ]
                writer.writerow(["Bead's datas"])
                writer.writerows(data_bead)
                writer.writerow([])
                writer.writerow(["Fitting's datas"])
                writer.writerows(data_fitting)
                writer.writerow([])
                writer.writerow(["Meric's results"])
                writer.writerows(data_metrics)
                writer.writerow([])
                writer.writerow([])

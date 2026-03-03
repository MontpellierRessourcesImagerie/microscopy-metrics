from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph,Table,TableStyle
import os
from jinja2 import Environment, FileSystemLoader
import csv

class Report_Generator(object):
    def __init__(self):
        self.output_dir = ""
        self.output_path = ""
        self.analysis_data = {}
        self.parameters_detection = {}
        self.parameters_acquisition = {}
        self.filtered_beads = []
        self.mean_SBR = 0.0

        self.pdf = None


    def get_active_path(self, index):
        """Utility function to return the current path of a given bead"""
        active_path = os.path.join(self.output_dir,f"bead_{index}")
        if not os.path.exists(active_path):
            os.makedirs(active_path)
        return active_path

    def draw_paragaph_on_pdf(self,textLines,x,y):
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet['Normal']
        full_text = "<br/>".join(textLines)
        p = Paragraph(full_text,normalStyle)
        p.wrapOn(self.pdf,500,100)
        p.drawOn(self.pdf,x,y)

    def draw_table_on_pdf(self,data):
        s = getSampleStyleSheet()
        s = s["BodyText"]
        s.wordWrap = 'CJK'
        data2 = [[Paragraph(cell, s) for cell in row] for row in data]
        t = Table(data=data2, colWidths=[80, 50, 50, 50])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        t.wrapOn(self.pdf, 0, 0)
        t.drawOn(self.pdf, 40, 600)

    def draw_single_bead_report_pdf(self,active_path,index):
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet['Normal']
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(150,770, f'Bead {index}')
        self.pdf.drawImage(os.path.join(active_path,"Localisation.png"),350,600,width=200,height=200,preserveAspectRatio=True)
        textLines = [
            f"centroid: {self.filtered_beads[self.analysis_data[index]["id"]]}",
            f"Signal to Background ratio: {self.analysis_data[index]["SBR"]:.4f}",
            f"Lateral asymmetry ratio: {self.analysis_data[index]["LAR"]:.4f}",
            f"Sphericity: {self.analysis_data[index]["sphericity"]:.4f}"
        ]
        text = self.pdf.beginText(40,750)
        text.setFont(normalStyle.fontName, normalStyle.fontSize)
        for line in textLines :
            text.textLine(line)
        self.pdf.drawText(text)
        data = [
            ["","Z","Y","X"],
            ["FWHM (µm)", f"{self.analysis_data[index]["FWHM"][0]:.4f}",f"{self.analysis_data[index]["FWHM"][1]:.4f}",f"{self.analysis_data[index]["FWHM"][2]:.4f}"],
            ["Uncertainty",f"{self.analysis_data[index]["uncertainty"][0][3]:.4f}",f"{self.analysis_data[index]["uncertainty"][1][3]:.4f}",f"{self.analysis_data[index]["uncertainty"][2][3]:.4f}"],
            ["Determination",f"{self.analysis_data[index]["determination"][0]:.4f}",f"{self.analysis_data[index]["determination"][1]:.4f}",f"{self.analysis_data[index]["determination"][2]:.4f}",]
        ]
        self.draw_table_on_pdf(data)
        self.pdf.drawImage(os.path.join(active_path,"YZ_view.png"),50,425,width=150,height=150,preserveAspectRatio=True)
        self.pdf.drawImage(os.path.join(active_path,"fit_curve_1D_X.png"),300,375,width=250,height=250,preserveAspectRatio=True)
        self.pdf.drawImage(os.path.join(active_path,"XZ_view.png"),50,225,width=150,height=150,preserveAspectRatio=True)
        self.pdf.drawImage(os.path.join(active_path,"fit_curve_1D_Y.png"),300,175,width=250,height=250,preserveAspectRatio=True)
        self.pdf.drawImage(os.path.join(active_path,"XY_view.png"),50,25,width=150,height=150,preserveAspectRatio=True)
        self.pdf.drawImage(os.path.join(active_path,"fit_curve_1D_Z.png"),300,-25,width=250,height=250,preserveAspectRatio=True)
        self.pdf.showPage()


    def generate_pdf_report(self,image_path):
        rois = [entry["ROI"] for entry in self.analysis_data]
        self.pdf = canvas.Canvas(self.output_path,pagesize=A4)
        self.pdf.setTitle("PSF analysis results")
        self.pdf.setFont("Helvetica-Bold", 36)
        self.pdf.drawCentredString(300,770, 'Analysis Results')
        textLines = [
            f"Image location: {image_path}",
            f"Identified beads: {len(self.filtered_beads)}",
            f"Extracted ROIs: {len(rois)}",
            f"Signal to background ratio: {self.mean_SBR:.2f}"
        ]
        self.draw_paragaph_on_pdf(textLines,40,680)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300,600, 'Acquisition parameters')
        textLines = [
            f"Pixel size: [{self.parameters_acquisition["PhysicSizeZ"]},{self.parameters_acquisition["PhysicSizeY"]},{self.parameters_acquisition["PhysicSizeX"]}]",
            f"Image shape: [{self.parameters_acquisition["ShapeZ"]},{self.parameters_acquisition["ShapeY"]},{self.parameters_acquisition["ShapeX"]}]",
            f"Microscope type: {self.parameters_acquisition["Microscope_type"]}",
            f"Emission wavelength: {self.parameters_acquisition["Emission_Wavelength"]}nm",
            f"Refractive index: {self.parameters_acquisition["Refractive_index"]}",
            f"Numerical aperture: {self.parameters_acquisition["Numerical_aperture"]}"
        ]
        self.draw_paragaph_on_pdf(textLines,40,500)
        self.pdf.setFont("Helvetica-Bold", 18)
        self.pdf.drawCentredString(300,400, 'Detection parameters')
        tools = ["peak_local_maxima", "blob_log", "blob_dog", "centroids"]
        textLines = [
            f"Detection method: {tools[self.parameters_detection['selected_tool']]}"
        ]
        if self.parameters_detection["selected_tool"] == 0:
            textLines.append(f"Minimal distance: {self.parameters_detection['Min_dist']}")
        else:
            textLines.append(f"Sigma: {self.parameters_detection['Sigma']}")
        textLines.extend([
            f"Bead size: {self.parameters_detection['theorical_bead_size']}",
            f"Crop factor: {self.parameters_detection['crop_factor']}"
        ])
        if self.parameters_detection["auto_threshold"]:
            textLines.append(f"Threshold tool: {self.parameters_detection['threshold_choice']}")
        else:
            textLines.append(f"Threshold relative: {self.parameters_detection['Rel_threshold']}")
        textLines.extend([
            f"Distance ring-bead: {self.parameters_detection['distance_annulus']}",
            f"Ring thickness: {self.parameters_detection['thickness_annulus']}"
        ])
        self.draw_paragaph_on_pdf(textLines,40,300)

        self.pdf.showPage()

        for i,_ in enumerate(rois):
            active_path = self.get_active_path(index=i)
            # Generating the HTML report
            self.draw_single_bead_report_pdf(active_path,i)

        self.pdf.save()


    def generate_html_report(self):
        """Function to automatically generate the report of a bead analysis in a html file based on a template"""
        for i,psf in enumerate(self.analysis_data):
            path = self.get_active_path(i)
            active_path = os.path.join(path,"PSF_analysis_result.html")
            template_dir = os.path.join(os.path.dirname(__file__),'res','template')
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template('report_template.html')
            data = {
                'title':i,
                'bead':self.filtered_beads[psf["id"]],
                'results':psf,
                'path':path
            }
            html_content = template.render(data)
            with open(active_path,'w') as f :
                f.write(html_content)

    def generate_csv_report(self,output_path):
        with open(output_path, mode='w',newline='') as file :
            writer = csv.writer(file)
            for i,psf in enumerate(self.analysis_data):
                writer.writerow([f"Bead {i}"])
                data_bead = [
                    ['','Z','Y','X'],
                    ['Centroid coordinates',f'{self.filtered_beads[i][0]}',f'{self.filtered_beads[i][1]}',f'{self.filtered_beads[i][2]}']
                ]
                data_fitting = [
                    ["","Z","Y","X"],
                    ["FWHM", f"{self.analysis_data[i]["FWHM"][0]}",f"{self.analysis_data[i]["FWHM"][1]}",f"{self.analysis_data[i]["FWHM"][2]}"],
                    ["Uncertainty",f"{self.analysis_data[i]["uncertainty"][0][3]}",f"{self.analysis_data[i]["uncertainty"][1][3]}",f"{self.analysis_data[i]["uncertainty"][2][3]}"],
                    ["Determination",f"{self.analysis_data[i]["determination"][0]}",f"{self.analysis_data[i]["determination"][1]}",f"{self.analysis_data[i]["determination"][2]}",]
                ]
                data_metrics = [
                    ["Metric","Value"],
                    ["Signal to background ratio",f"{self.analysis_data[i]["SBR"]}"],
                    ["Lateral asymmetry ratio",f"{self.analysis_data[i]["LAR"]}"],
                    ["Sphericity",f"{self.analysis_data[i]["sphericity"]}"]
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

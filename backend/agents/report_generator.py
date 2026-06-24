from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path
import os

def clinical_interpretation(prediction, risk, size_cm):
    findings = []
    
    pred_lower = prediction.lower() if prediction else ""
    if "meningioma" in pred_lower:
        findings += [
            "Well-defined extra-axial lesion noted.",
            "Imaging features suggest benign meningioma."
        ]
    elif "glioma" in pred_lower:
        findings += [
            "Intra-axial infiltrative lesion identified.",
            "Findings suspicious for primary glial tumor."
        ]
    elif "pituitary" in pred_lower:
        findings += [
            "Sellar/suprasellar mass observed.",
            "Features suggest pituitary adenoma."
        ]
    else:
        findings.append("No definite intracranial mass lesion detected.")

    findings.append(f"Estimated tumor size: {size_cm:.2f} cm.")
    findings.append(f"Clinical risk category: {risk}")

    if risk == "High":
        impression = "Large tumor with significant clinical risk requiring urgent medical evaluation."
    elif risk == "Medium":
        impression = "Moderate tumor presence requiring specialist consultation."
    else:
        impression = "Small lesion with low immediate clinical concern."

    suggestions = [
        "Correlate findings with clinical symptoms",
        "Recommend contrast-enhanced MRI for detailed evaluation",
        "Consult neurosurgeon or neurologist for treatment planning"
    ]

    return findings, impression, suggestions


def generate_pdf(patient, analysis, img_path):
    backend_dir = Path(__file__).resolve().parent.parent
    static_dir = backend_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = static_dir / "Radiology_Report.pdf"

    doc = SimpleDocTemplate(
        str(file_path),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=20
    )

    story = []

    section_style = ParagraphStyle("section", fontSize=11, textColor=colors.HexColor("#0B5394"))
    normal_style = ParagraphStyle("normal", fontSize=9)

    header = Table([["AI NEURO IMAGING CENTER"]], colWidths=[7.5 * inch])
    header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0B5394")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8)
    ]))
    story.append(header)
    story.append(Spacer(1,6))
    story.append(Paragraph("MRI BRAIN RADIOLOGY REPORT", section_style))
    story.append(Spacer(1,10))

    story.append(Paragraph(f"<b>Doctor:</b> {patient.get('doctor', 'Unknown')}", normal_style))
    story.append(Spacer(1,6))

    story.append(Paragraph("PATIENT DETAILS", section_style))
    story.append(Spacer(1,6))

    info = [
        ["Patient Name", patient.get("name", "Unknown")],
        ["Age", patient.get("age", "Unknown")],
        ["Blood Group", patient.get("blood_group", "Unknown")],
        ["Gender", patient.get("gender", "Unknown")],
        ["Date", datetime.now().strftime("%d-%m-%Y")]
    ]

    table = Table(info, colWidths=[2.2*inch, 4.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#D9E1F2")),
        ('GRID',(0,0),(-1,-1),0.4,colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1,10))

    story.append(Paragraph("AI MEASUREMENTS", section_style))
    story.append(Paragraph(f"<b>Tumor Type:</b> {analysis.get('prediction', 'Unknown')}", normal_style))
    story.append(Paragraph(f"<b>Model Confidence:</b> {analysis.get('confidence', 0)*100:.2f}%", normal_style))
    story.append(Paragraph(f"<b>Risk Category:</b> {analysis.get('risk_level', 'Unknown')}", normal_style))
    story.append(Paragraph(f"<b>Estimated Tumor Size:</b> {analysis.get('tumor_size_cm', 0):.2f} cm", normal_style))
    story.append(Spacer(1,8))

    if img_path and os.path.exists(img_path):
        story.append(Paragraph("Tumor Localization", section_style))
        img = Image(img_path, width=2.4 * inch, height=2.4 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1,6))

    findings, impression, suggestions = clinical_interpretation(
        analysis.get("prediction", ""),
        analysis.get("risk_level", ""),
        analysis.get("tumor_size_cm", 0)
    )

    story.append(Paragraph("KEY FINDINGS", section_style))
    for f in findings:
        story.append(Paragraph("• " + f, normal_style))
    story.append(Spacer(1,6))

    story.append(Paragraph("IMPRESSION", section_style))
    story.append(Paragraph(impression, normal_style))
    story.append(Spacer(1,6))

    story.append(Paragraph("SUGGESTIONS", section_style))
    for s in suggestions:
        story.append(Paragraph("• " + s, normal_style))

    doc.build(story)
    return str(file_path)

import os
import base64
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_table_row(table, key, value, bold_key=True):
    row = table.add_row().cells
    
    # Key cell
    key_run = row[0].paragraphs[0].add_run(key)
    if bold_key:
        key_run.bold = True
    row[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    # Value cell
    row[1].text = str(value) if value else "N/A"
    
def set_col_widths(table, widths):
    for row in table.rows:
        for idx, width in enumerate(widths):
            row.cells[idx].width = width

def export_to_docx(image_b64: str, node_details: dict) -> BytesIO:
    """
    Generate a Word document containing the architecture flowchart image and
    detailed tables for each module.
    """
    doc = Document()
    
    # Set default font for Chinese characters support 
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Microsoft YaHei'
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

    # Title
    title = doc.add_heading("System Architecture Plan", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add Flowchart Image
    doc.add_heading("1. Architecture Flowchart", level=1)
    
    if image_b64:
        # Remove data:image/png;base64, prefix if exists
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
            
        try:
            image_data = base64.b64decode(image_b64)
            image_stream = BytesIO(image_data)
            
            # Add image to document, scale down if it is too large
            doc.add_picture(image_stream, width=Inches(6.0))
            last_paragraph = doc.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception as e:
            doc.add_paragraph(f"[Error rendering image: {str(e)}]")
    else:
        doc.add_paragraph("[No flowchart image provided]")

    doc.add_page_break()

    # Add Node Details
    doc.add_heading("2. Module Details", level=1)

    # Normalize node details (it can be a dict or a list)
    if isinstance(node_details, dict):
        modules = list(node_details.values())
    else:
         modules = node_details
    
    if not modules:
        doc.add_paragraph("No detailed module information available.")
    
    for idx, mod in enumerate(modules):
         mod_name = mod.get("name", f"Module {idx+1}")
         doc.add_heading(f"2.{idx+1} {mod_name}", level=2)
         
         # Create a table for the module properties
         table = doc.add_table(rows=0, cols=2)
         table.style = 'Table Grid'
         
         # Add standard fields
         add_table_row(table, "Brief Description", mod.get("brief_description", ""))
         add_table_row(table, "Function", mod.get("function", ""))
         add_table_row(table, "Technical Methods", mod.get("technical_methods", ""))
         
         if "role" in mod:
                add_table_row(table, "Role", mod.get("role"))
         if "references" in mod:
                add_table_row(table, "References", mod.get("references"))
         
         set_col_widths(table, [Inches(1.5), Inches(4.5)])
         doc.add_paragraph("\n")
         
         # Add sub-nodes if exist
         sub_nodes = mod.get("sub_nodes", [])
         if sub_nodes:
             doc.add_heading(f"Sub-nodes for {mod_name}", level=3)
             for sn_idx, sub in enumerate(sub_nodes):
                 sn_name = sub.get("name", "Unknown Sub-node")
                 p = doc.add_paragraph()
                 p.add_run(f"• {sn_name}").bold = True
                 
                 sn_table = doc.add_table(rows=0, cols=2)
                 sn_table.style = 'Light Shading Accent 1'
                 
                 add_table_row(sn_table, "Function", sub.get("function", ""))
                 add_table_row(sn_table, "Technical Methods", sub.get("technical_methods", ""))
                 set_col_widths(sn_table, [Inches(1.5), Inches(4.5)])
                 doc.add_paragraph("\n")

    # Save document to a BytesIO stream
    doc_stream = BytesIO()
    doc.save(doc_stream)
    doc_stream.seek(0)
    
    return doc_stream

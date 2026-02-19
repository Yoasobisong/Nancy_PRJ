import PyPDF2
import sys

def extract_pdf_text(pdf_path, output_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- PAGE {i+1} ---\n{page_text}"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages from {pdf_path}")

extract_pdf_text(r"d:\PRJ\Nancy_PRJ\doc\2Human-robot_ficial_coexpression.pdf", 
                 r"d:\PRJ\Nancy_PRJ\doc\paper2_text.txt")
extract_pdf_text(r"d:\PRJ\Nancy_PRJ\doc\4Learning_realistic_lip_motions.pdf",
                 r"d:\PRJ\Nancy_PRJ\doc\paper4_text.txt")

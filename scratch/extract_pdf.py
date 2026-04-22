import PyPDF2
import os

pdf_path = r'c:\Users\ADMIN\Desktop\Var-CNN\paper\1802.10215v2.pdf'
out_path = r'c:\Users\ADMIN\Desktop\Var-CNN\scratch\paper_text.txt'

if os.path.exists(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(text)
    print(f"Extracted {len(text)} characters to {out_path}.")
else:
    print(f"File not found: {pdf_path}")

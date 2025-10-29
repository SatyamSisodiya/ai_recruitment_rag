# ingestion/ocr.py
import pdfplumber
import pytesseract
from PIL import Image
import io
import os
from typing import Optional

from utils import normalize_text

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
                else:
                    # fallback to image-based OCR on page image
                    im = page.to_image(resolution=300).original
                    txt2 = pytesseract.image_to_string(im)
                    text_parts.append(txt2)
    except Exception as e:
        # fallback to pure OCR if pdfplumber fails
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(path, dpi=300)
            for page in pages:
                txt = pytesseract.image_to_string(page)
                text_parts.append(txt)
        except Exception as e2:
            raise e2
    return normalize_text("\n".join(text_parts))

def extract_text_from_image(path: str) -> str:
    im = Image.open(path)
    text = pytesseract.image_to_string(im)
    return normalize_text(text)

def extract_text_from_docx(path: str) -> str:
    import docx
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return normalize_text("\n".join(paragraphs))

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        return extract_text_from_pdf(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        return extract_text_from_image(path)
    elif ext in [".docx"]:
        return extract_text_from_docx(path)
    elif ext in [".txt"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return normalize_text(f.read())
    else:
        raise ValueError(f"Unsupported file type: {ext}")

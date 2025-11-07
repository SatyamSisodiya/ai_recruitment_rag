# ingestion/ocr.py
import pdfplumber
from PIL import Image
import io
import os
from typing import Optional

from utils import normalize_text

# Guarded import for Tesseract OCR; not available on many free hosts
try:
    import pytesseract  # type: ignore
    _OCR_AVAILABLE = True
except Exception:
    pytesseract = None  # type: ignore
    _OCR_AVAILABLE = False

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
                else:
                    # If OCR is available, attempt image OCR; otherwise, skip gracefully
                    if _OCR_AVAILABLE:
                        try:
                            im = page.to_image(resolution=300).original
                            txt2 = pytesseract.image_to_string(im)
                            text_parts.append(txt2)
                        except Exception:
                            # Skip OCR on failure; continue
                            pass
    except Exception as e:
        # Fallback to pure OCR only if OCR stack is available; else return what we have
        if _OCR_AVAILABLE:
            try:
                from pdf2image import convert_from_path  # Optional dependency
                pages = convert_from_path(path, dpi=300)
                for page in pages:
                    try:
                        txt = pytesseract.image_to_string(page)
                        text_parts.append(txt)
                    except Exception:
                        pass
            except Exception:
                # No pdf2image or OCR failed; proceed with whatever was extracted
                pass
        # If OCR not available, we simply proceed with any collected text (likely none)
    return normalize_text("\n".join(text_parts))

def extract_text_from_image(path: str) -> str:
    # If OCR unavailable, return empty string to degrade gracefully
    if not _OCR_AVAILABLE:
        return ""
    im = Image.open(path)
    try:
        text = pytesseract.image_to_string(im)
    except Exception:
        text = ""
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

def ocr_available() -> bool:
    """Return True if Tesseract OCR is available in the current environment."""
    return _OCR_AVAILABLE

import fitz  # PyMuPDF
import re
import os
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.

    """

    doc = fitz.open(pdf_path)
    full_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")  # Use "text" for raw or "blocks" for more structure
        cleaned = clean_text(text)
        full_text.append(cleaned)

    return "\n\n".join(full_text)

def clean_text(text: str) -> str:
    """
    Clean the extracted text from a PDF file.

    Args:
        text (str): The extracted text from the PDF file.

    Returns:
        str: The cleaned text from the PDF file.

    """

    # Remove page numbers and form feeds
    text = re.sub(r'Page\s*\d+|\f', '', text)

    # Remove multiple newlines and normalize spacing
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Remove extra whitespace at start/end of lines
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)

    # Ensure paragraphs are properly separated
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs)

def get_pdf_files(directory: str) -> list[str]:
    """
    Get all PDF files in the specified directory.

    Args:
        directory (str): The directory to search for PDF files.

    Returns:
        list[str]: A list of paths to the PDF files.

    """

    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

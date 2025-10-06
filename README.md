# BTP-Sub-Ans-Evaluation-System

Grading Answer Sheets

This project extracts student answers from scanned answer sheets, computes semantic similarity against a model answer using Sentence-BERT, checks grammar and readability, and grades each answer.

Contents:
- `main.py` - main script that runs OCR, grading, and visualizes results.
- `datasets/` - sample image datasets (excluded from VCS by default in `.gitignore`).
- `sample_images/` - sample images used for testing.

Getting started:
1. Install Python 3.8+.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run `python main.py`.

Notes:
- Ensure Tesseract OCR is installed and `pytesseract.pytesseract.tesseract_cmd` points to the correct path in `main.py`.
- Large datasets are excluded by default; add them manually to the repo if needed.
<<<<<<< HEAD
# BTP-Sub-Ans-Evaluation-System
=======
# Grading Answer Sheets

This project extracts student answers from scanned answer sheets, computes semantic similarity against a model answer using Sentence-BERT, checks grammar and readability, and grades each answer.

Contents:
- `main.py` - main script that runs OCR, grading, and visualizes results.
- `datasets/` - sample image datasets (excluded from VCS by default in `.gitignore`).
- `sample_images/` - sample images used for testing.

Getting started:
1. Install Python 3.8+.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run `python main.py`.

Notes:
- Ensure Tesseract OCR is installed and `pytesseract.pytesseract.tesseract_cmd` points to the correct path in `main.py`.
- Large datasets are excluded by default; add them manually to the repo if needed.
>>>>>>> e2daa54 (Initial commit: project scaffold and requirements)

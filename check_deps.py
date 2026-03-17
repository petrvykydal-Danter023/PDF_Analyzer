import pytesseract
import sys
import os

print(f"Platform: {sys.platform}")
print(f"Python: {sys.version}")

try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
    print(f"Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")
except Exception as e:
    print(f"Error finding Tesseract: {e}")
    # Try manual check
    win_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for p in win_paths:
        if os.path.exists(p):
            print(f"Found Tesseract manually at: {p}")
            break
    else:
        print("Tesseract not found in common Windows locations.")

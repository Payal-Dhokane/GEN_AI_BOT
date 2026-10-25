import pytesseract
from PIL import Image

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_image(image_path):
    """
    Extract readable text from an image using OCR.
    """
    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text.strip()


def analyze_image(image_path):
    """
    Basic image evidence extraction.
    Currently extracts OCR text.
    """

    extracted_text = extract_text_from_image(image_path)

    return {
        "image_path": image_path,
        "extracted_text": extracted_text,
        "has_text": bool(extracted_text)
    }


if __name__ == "__main__":

    result = analyze_image("Zenia.jpeg")

    print("\nImage Analysis Result")
    print("---------------------")

    print("Has readable text:", result["has_text"])
    print("Extracted text:", result["extracted_text"])
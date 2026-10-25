
from image_analyzer import extract_text_from_image
from visual_analyzer import analyze_visual_content
from reasoning_engine import reason_over_evidence


def analyze_multimodal_input(image_path, conversation_history=None):
    extracted_text = extract_text_from_image(image_path)
    visual_results = analyze_visual_content(image_path, top_k=5)

    top_prediction = visual_results[0] if visual_results else None

    ambiguous = False
    if len(visual_results) >= 2:
        ambiguous = (
            visual_results[0]["confidence"]
            - visual_results[1]["confidence"]
        ) < 0.25

    evidence = {
        "ocr_text": extracted_text,
        "visual_predictions": visual_results,
        "top_prediction": top_prediction,
        "ambiguous": ambiguous
    }

    evidence["reasoning"] = reason_over_evidence(
        evidence,
        conversation_history=conversation_history
    )

    return evidence


if __name__ == "__main__":
    result = analyze_multimodal_input("Zenia.jpeg")

    print("\nMULTIMODAL IMAGE ANALYSIS")
    print("=========================")
    print("\nOCR Evidence:")
    print(result["ocr_text"] or "No readable text detected.")

    print("\nVisual Evidence:")
    for item in result["visual_predictions"]:
        print(f"- {item['label']} ({item['confidence']})")

    print("\nAmbiguity Detected:", result["ambiguous"])

    print("\nEvidence-Based Reasoning:")
    print(result["reasoning"])

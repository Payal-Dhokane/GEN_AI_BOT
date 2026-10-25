from transformers import pipeline


# Lightweight image classification model
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)


def analyze_visual_content(image_path, top_k=5):
    """
    Analyze the visual content of an image
    and return the most likely objects/classes.
    """

    results = classifier(image_path, top_k=top_k)

    visual_evidence = []

    for result in results:
        visual_evidence.append({
            "label": result["label"],
            "confidence": round(result["score"], 3)
        })

    return visual_evidence


if __name__ == "__main__":

    print("\nVisual Analysis")
    print("----------------")

    results = analyze_visual_content("Zenia.jpeg")

    for item in results:
        print(
            f"{item['label']} "
            f"(confidence: {item['confidence']})"
        )
from transformers import pipeline

# Load 3-class sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def analyze_sentiment(text):
    """
    Returns:
    {
        "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
        "score": confidence
    }
    """
    result = sentiment_pipeline(text)[0]

    return {
        "label": result["label"].upper(),
        "score": round(result["score"], 3)
    }


if __name__ == "__main__":
    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            break

        print(analyze_sentiment(text))
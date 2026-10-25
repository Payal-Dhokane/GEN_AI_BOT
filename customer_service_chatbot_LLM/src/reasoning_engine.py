
from langchain_ollama import ChatOllama


def reason_over_evidence(evidence, conversation_history=None):
    conversation_history = conversation_history or []

    model = ChatOllama(
        model="llama3.2:latest",
        temperature=0
    )

    history_text = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in conversation_history[-6:]
    )

    prompt = f"""
You are a careful multimodal assistant.

Analyze the supplied evidence without inventing visual details.
Distinguish likely conclusions from confirmed facts.
If evidence is uncertain, clearly explain the uncertainty.

Previous conversation:
{history_text or "No previous conversation."}

Image evidence:
{evidence}

Return:
1. Interpretation
2. Supporting evidence
3. Uncertainty or missing information
"""

    response = model.invoke(prompt)
    return response.content


if __name__ == "__main__":
    sample_evidence = {
        "extracted_text": "",
        "visual_predictions": [
            {"label": "pot, flowerpot", "score": 0.519},
            {"label": "daisy", "score": 0.330},
            {"label": "vase", "score": 0.055}
        ],
        "ambiguous": True
    }

    result = reason_over_evidence(sample_evidence)
    print("\nREASONING ENGINE OUTPUT")
    print("=======================")
    print(result)

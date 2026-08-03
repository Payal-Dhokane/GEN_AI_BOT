import re

# Disease keywords
DISEASES = [
    "diabetes", "asthma", "cancer", "migraine", "arthritis",
    "pneumonia", "covid", "tuberculosis", "anemia", "hypertension"
]

# Symptom keywords
SYMPTOMS = [
    "fever", "headache", "cough", "pain", "fatigue",
    "vomiting", "nausea", "dizziness", "rash", "swelling"
]

# Treatment keywords
TREATMENTS = [
    "treatment", "medicine", "drug", "therapy", "surgery",
    "insulin", "antibiotics", "vaccine", "chemotherapy", "radiation"
]


def detect_medical_entities(text):

    text = text.lower()

    diseases = [
        word for word in DISEASES
        if re.search(rf"\b{re.escape(word)}\b", text)
    ]

    symptoms = [
        word for word in SYMPTOMS
        if re.search(rf"\b{re.escape(word)}\b", text)
    ]

    treatments = [
        word for word in TREATMENTS
        if re.search(rf"\b{re.escape(word)}\b", text)
    ]

    return {
        "Diseases": diseases,
        "Symptoms": symptoms,
        "Treatments": treatments,
    }
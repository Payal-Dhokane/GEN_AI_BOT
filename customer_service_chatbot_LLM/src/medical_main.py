import streamlit as st
from medical_helper import get_medical_chain
from medical_entity import detect_medical_entities

# Page Configuration
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Q&A Chatbot")
st.caption("Powered by MedQuAD, LangChain, Gemini & FAISS")

st.divider()

question = st.text_input("Ask your medical question:")

if question:

    entities = detect_medical_entities(question)

    st.subheader("🩺 Detected Medical Entities")

    if any(entities.values()):

        if entities["Diseases"]:
            st.write("**Disease:**", ", ".join(entities["Diseases"]))

        if entities["Symptoms"]:
            st.write("**Symptom:**", ", ".join(entities["Symptoms"]))

        if entities["Treatments"]:
            st.write("**Treatment:**", ", ".join(entities["Treatments"]))

    else:
        st.write("No medical entities detected.")

if question:

    with st.spinner("Searching medical knowledge..."):
        chain = get_medical_chain()
        response = chain.invoke({"query": question})

    answer = response["result"]

    st.subheader("📖 Medical Answer")
    st.write(answer)
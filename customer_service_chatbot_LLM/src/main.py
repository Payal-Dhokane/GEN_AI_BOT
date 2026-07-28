import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from sentiment import analyze_sentiment


def get_response_prefix(label):
    if label == "POSITIVE":
        return "😊 Thank you for your positive feedback! I'm happy to assist you."

    elif label == "NEGATIVE":
        return "😔 I'm sorry you're facing this issue. I'll do my best to help resolve it."

    else:
        return "😐 Thank you for your question. Here's the information you requested."



# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Customer Service Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Service Chatbot")
st.caption("Powered by LangChain, Gemini, FAISS & RoBERTa Sentiment Analysis")

st.divider()

# Create Knowledge Base
if st.button("Create Knowledgebase"):
    with st.spinner("Creating Knowledge Base..."):
        create_vector_db()
    st.success("Knowledge Base created successfully!")

# User Input
question = st.text_input("Question:")

# Chatbot
if question:

    # Sentiment Analysis
    sentiment = analyze_sentiment(question)
    label = sentiment["label"]
    score = sentiment["score"]

    # Generate Response
    with st.spinner("Generating response..."):
        chain = get_qa_chain()
        response = chain.invoke({"query": question})

    # Sentiment Section
    st.subheader("📊 Sentiment Analysis")

    if label == "POSITIVE":
        st.success(f"😊 Positive ({score:.2%})")
    elif label == "NEGATIVE":
        st.error(f"😔 Negative ({score:.2%})")
    else:
        st.info(f"😐 Neutral ({score:.2%})")

    # Customer Support Message
    st.subheader("💬 Customer Support")
    st.write(get_response_prefix(label))

    
    # Knowledge Base Response

    answer = response["result"].strip()

    if answer.lower() == "yes":
        answer = "Yes, we do provide this service."

    elif answer.lower() == "no":
        answer = "No, this service is currently not available."

    st.subheader("📖 Knowledge Base Response")
    st.write(answer)
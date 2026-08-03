import os
from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

medical_index = "medical_faiss_index"


def create_medical_vector_db():

    print("Loading medical dataset...")

    loader = CSVLoader(
        file_path="../dataset/medical_dataset.csv",
        source_column="prompt",
        encoding="utf-8",
    )

    docs = loader.load()

    print(f"Loaded {len(docs)} documents")

    print("Creating embeddings (this may take several minutes)...")

    vectordb = FAISS.from_documents(docs, embeddings)

    print("Saving FAISS index...")

    vectordb.save_local(medical_index)

    print("✅ Medical Vector Database Created Successfully!")


def get_medical_chain():

    vectordb = FAISS.load_local(
        medical_index,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    prompt = PromptTemplate(
        template="""
You are a medical question answering assistant.

Answer ONLY from the provided medical context.

If the answer is not available, reply:

"I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


if __name__ == "__main__":
    create_medical_vector_db()
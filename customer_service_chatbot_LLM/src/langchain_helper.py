import os
from dotenv import load_dotenv

import langchain_classic
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings


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


vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(
    file_path="../dataset/dataset.csv",
    source_column="prompt",
)
    docs = loader.load()

    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever()

    prompt = PromptTemplate(
        template="""
Answer the question only from the provided context.
If the answer is not present, reply:
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
    create_vector_db()
    print("Vector database created successfully!")
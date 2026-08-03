import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS Folder
vectordb_file_path = "faiss_index"


# ---------------------------------------------------
# Create Knowledge Base from Default Dataset
# ---------------------------------------------------
def create_vector_db():
    loader = CSVLoader(
        file_path="../dataset/dataset.csv",
        source_column="prompt",
    )

    docs = loader.load()

    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(vectordb_file_path)


# Update Existing Knowledge Base

def update_vector_db(new_csv_path):
    loader = CSVLoader(
        file_path=new_csv_path,
        source_column="prompt",
    )

    new_docs = loader.load()

    # If FAISS already exists
    if Path(vectordb_file_path).exists():

        vectordb = FAISS.load_local(
            vectordb_file_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Existing prompts
        existing_prompts = {
            doc.page_content.strip()
            for doc in vectordb.docstore._dict.values()
        }

        # Keep only new documents
        unique_docs = [
            doc for doc in new_docs
            if doc.page_content.strip() not in existing_prompts
        ]

        if unique_docs:
            vectordb.add_documents(unique_docs)

        vectordb.save_local(vectordb_file_path)

        return len(unique_docs)

    else:

        vectordb = FAISS.from_documents(new_docs, embeddings)
        vectordb.save_local(vectordb_file_path)

        return len(new_docs)

# ---------------------------------------------------
# Question Answering Chain
# ---------------------------------------------------
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


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    create_vector_db()
    print("Vector database created successfully!")
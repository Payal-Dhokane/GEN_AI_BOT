from langchain_helper import update_vector_db

count = update_vector_db("../dataset/dataset.csv")

print(f"Successfully added {count} documents to the knowledge base.")
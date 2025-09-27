import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def save_faiss_index(vectorstore, save_path="faiss_index"):
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

def load_faiss_index(openai_key, save_path="faiss_index"):
    embedding = OpenAIEmbeddings(openai_api_key=openai_key)
    return FAISS.load_local(
        save_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
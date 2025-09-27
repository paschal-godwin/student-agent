import streamlit as st
from dotenv import load_dotenv
import os
from load_and_split import process_pdf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from utils import save_faiss_index, load_faiss_index
from summarizer import generate_summary
from flashcard import generate_flashcard

# --- Load environment and configs ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

BOOKS_DIR = "books"         # Permanent PDFs
INDEX_DIR = "faiss_index"   # FAISS store path

st.set_page_config(page_title="📚 Student Study Agent", layout="wide")
st.title("📚 Student Study Agent")

# --- Ensure folders exist ---
os.makedirs(BOOKS_DIR, exist_ok=True)

# --- Initialize embeddings ---
embedding = OpenAIEmbeddings(
    openai_api_key=openai_key,
)

# --- Load or create vectorstore ---
vectorstore = None

if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
    with st.spinner("🔄 Loading your knowledge base..."):
        try:
            vectorstore = load_faiss_index(openai_key, INDEX_DIR)
            st.success("✅ Loaded existing study memory.")
        except Exception as e:
            st.error(f"❌ Failed to load FAISS index: {e}")
else:
    with st.spinner("📖 Creating memory from books/ folder..."):
        all_docs = []
        for filename in os.listdir(BOOKS_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(BOOKS_DIR, filename)
                st.write(f"Processing: {filename}")
                docs = process_pdf(file_path)
                all_docs.extend(docs)

        if all_docs:
            vectorstore = FAISS.from_documents(all_docs, embedding)
            save_faiss_index(vectorstore, INDEX_DIR)
            st.success("✅ Knowledge base created from books.")
        else:
            st.warning("⚠️ No PDFs found in books/ folder.")
            vectorstore = None

# --- PDF Upload (temporary) ---
uploaded_file = st.file_uploader("📄 Upload a study PDF (temporary)", type="pdf")

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
            docs, load_time = process_pdf(uploaded_file)
            if docs:
                vectorstore.add_documents(docs)
                st.success(f"✅ {len(docs)} chunks loaded in {load_time:.2f} seconds.")
            else:
                st.error("❌ Failed to extract content from uploaded file.")


# --- Q&A Section ---
if vectorstore:
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = st.text_input("🧠 Ask a study question")

    if question:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(question)
            st.markdown("### ✅ Answer")
            st.write(response["result"])
else:
    st.info("Upload a PDF or place one in the 'books/' folder to get started.")


st.markdown("---")
# --- Comprehensive Summarizer (drop-in replacement) ---
st.subheader("📝 Summarize a Topic")

generate_summary(vectorstore)


st.markdown("---")
# --- Flashcards (Anki-style reveal) ---
st.subheader("🧠 Generate Flashcards")

generate_flashcard(vectorstore)

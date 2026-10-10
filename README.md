🧑‍🎓 Student Agent

An AI-powered study assistant that helps students summarize textbooks, generate flashcards, and answer questions from study materials. Built with Streamlit, LangChain, FAISS, and OpenAI.

✨ Features

📘 Topic Summarizer → generate detailed, structured summaries from uploaded PDFs or preloaded textbooks.

🧠 Flashcard Generator → creates Anki-style Q&A cards from any topic for active recall practice.

🔍 Q&A Assistant → ask natural language questions and get concise answers with references to source material.

📑 Source Highlighting → shows where answers came from (filename, page, and snippet).

🔐 API-Key Support → option for users to provide their own OpenAI API key to reduce hosting costs.

🚀 Demo

👉 Live Demo on Hugging Face Spaces
 


📂 Project Structure
student-agent/
│── app.py                # main Streamlit app
│── summarizer.py          # summary logic
│── flashcards.py          # flashcard logic
│── qa.py                  # question answering logic
│── load_and_split.py      # PDF processing & chunking
│── requirements.txt       # dependencies
│── .gitignore
│── README.md
│── /docs                  # screenshots, diagrams

⚙️ Installation

Clone the repo

git clone https://github.com/paschal-godwin/student-agent.git
cd student-agent


Create a virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Set environment variables
Create a .env file in the root:

OPENAI_API_KEY=your_api_key_here

▶️ Run the app
streamlit run app.py


📚 How It Works

PDFs are uploaded or loaded from a local book directory.

Documents are split into chunks with metadata (filename + page).

Chunks are embedded with OpenAI embeddings into a FAISS vectorstore.

Queries (summarize / flashcards / Q&A) are answered using retrieval-augmented generation (RAG).

Outputs are displayed in a clean, student-friendly Streamlit UI.



🛡️ License

MIT License — feel free to use and adapt.

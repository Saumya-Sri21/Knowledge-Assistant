# Knowledge-Assistant 

A **Streamlit-based RAG (Retrieval-Augmented Generation) app** using Google Gemini, LangChain and FAISS.  
Ask questions strictly from preloaded PDF documents.

# 📄 GenAI RAG Assistant

A Streamlit-based **Retrieval-Augmented Generation (RAG)** app that answers questions **strictly from PDF documents** using Google Gemini, LangChain, FAISS, and HuggingFace Embeddings.

---

## Features

### Document Loading
- Loads multiple PDF files from a folder

### Text Chunking
- Splits large documents into small chunks

### Embeddings
- Uses HuggingFace `all-MiniLM-L6-v2`

### Vector Database
- Stores embeddings in **FAISS**

### LLM Integration
- Uses **Google Gemini** via LangChain

### Context-Only Answers
- Answers strictly from provided documents
- If answer not found:  
  `"I don't know based on the provided documents."`

### Source Visibility
- Shows source PDF name and page number
---

## Tech Stack
- **LLM:** Google-Gemini 
- **Framework:** LangChain
- **Vector Database:** FAISS
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit

##  Prerequisites
- Python version - 3.10+
- Google Gemini API Key

##  Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Lakshya0018UP/GENAI
   cd Knowledge-Assistant

## 2. Create `.env` file

```GOOGLE_API_KEY=your_api_key_here```

## 3. Install Dependencies

```pip install -r requirements.txt```

## 4.Place your pdf in the data folderand run 

```uv run ingest.py```

## 5. Run the Application

```streamlit run app.py```

## Video Demo
[Demo Video Link]()

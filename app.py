# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import RetrievalQA
# # from langchain.prompts import PromptTemplate
# from langchain_classic.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# load_dotenv()

# st.set_page_config(page_title="GenAI RAG Assistant", layout="wide")

# st.title("📄Knowledge Assistant")
# st.write("Upload documents and ask questions strictly from context.")

# # ---------- Upload PDFs ----------
# uploaded_files = st.file_uploader(
#     "Upload PDF documents",
#     type=["pdf"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     os.makedirs("temp_docs", exist_ok=True)

#     documents = []
#     for file in uploaded_files:
#         file_path = f"temp_docs/{file.name}"
#         with open(file_path, "wb") as f:
#             f.write(file.getbuffer())

#         loader = PyMuPDFLoader(file_path)
#         documents.extend(loader.load())

#     # Chunking
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,
#         chunk_overlap=150
#     )
#     chunks = splitter.split_documents(documents)

#     # Embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

#     # FAISS Vector Store
#     vector_db = FAISS.from_documents(chunks, embeddings)

#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})

#     # LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0
#     )

#     # Prompt
#     template = """You are a specialized Knowledge Assistant.
#     Answer strictly from the provided context.

#     Rules:
#     - Use only the given context.
#     - If answer is not present, say "I don't know based on the provided documents."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )

#     st.success("✅ Documents processed successfully!")

#     # ---------- Q&A ----------
#     query = st.text_input("Ask a question from the uploaded documents:")

#     if query:
#         response = qa_chain.invoke({"query": query})

#         st.subheader("📌 Answer")
#         st.write(response["result"])

#         st.subheader("📚 Sources")
#         for doc in response["source_documents"]:
#             st.write(
#                 f"- {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', 'N/A')}"
#             )
# else:
#     st.info("Please upload at least one PDF to start.")


import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="GenAI RAG Assistant", layout="wide")
st.title("📄 Knowledge Assistant")
st.write("Answering from the preloaded documents")

# Load Documents
DOCS_PATH = "data"   
if not os.path.exists(DOCS_PATH):
    st.error("❌ 'docs' folder not found. Please add documents.")
    st.stop()

loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)

documents = loader.load()

if not documents:
    st.warning("⚠️ No PDF files found in docs folder.")
    st.stop()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# FAISS Vector Store
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Prompt
template = """
You are a specialized Knowledge Assistant.
Answer strictly from the provided context.

Rules:
- Use only the given context.
- If the answer is not present, say:
  "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

st.success("✅ Documents loaded and indexed successfully!")

# Q&A
query = st.text_input("Ask your question:")

if query:
    response = qa_chain.invoke({"query": query})

    st.subheader("📌 Answer")
    st.write(response["result"])

    st.subheader("📚 Sources")
    for doc in response["source_documents"]:
        st.write(
            f"- {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', 'N/A')}"
        )

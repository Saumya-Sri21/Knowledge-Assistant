from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def ingestion_service():
    print("Ingestion started...")

    #Load PDF Files
    pdf_loader = DirectoryLoader(
        "./data",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    pdf_documents = pdf_loader.load()
    print("PDF files loaded:", len(pdf_documents))

    #Load TXT Files
    txt_loader = DirectoryLoader(
        "./data",
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
        show_progress=True
    )
    txt_documents = txt_loader.load()
    print("TXT files loaded:", len(txt_documents))

    #Load MD Files
    md_loader = DirectoryLoader(
        "./data",
        glob="**/*.md",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
        show_progress=True
    )
    md_documents = md_loader.load()
    print("MD files loaded:", len(md_documents))

    all_documents = pdf_documents + txt_documents + md_documents
   
    if len(all_documents) == 0:
        print("No PDF, TXT, or MD files found!")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    document_chunks = text_splitter.split_documents(all_documents)
    print("Chunks created:", len(document_chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        document_chunks,
        embeddings
    )

    vector_db.save_local("./faiss_index")
    print("FAISS index saved successfully!")


if __name__ == "__main__":
    ingestion_service()

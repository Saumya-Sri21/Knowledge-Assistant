from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingestion_service():
    print("Ingestion started")

    dir_load = DirectoryLoader(
        "./data",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True           #just to show loading in console
    )

    pdf_documents = dir_load.load()
    print("PDFs loaded:", len(pdf_documents))

    if len(pdf_documents) == 0:
        print("No PDFs found")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    document_chunks = text_splitter.split_documents(pdf_documents)
    print("Chunks created:", len(document_chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        document_chunks,
        embeddings
    )

    vector_db.save_local("./faiss_index")
    print("FAISS index saved successfully")

if __name__ == "__main__":
    ingestion_service()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load FAISS index
    vector_db = FAISS.load_local(
        "./faiss_index",
        embeddings,
        allow_dangerous_deserialization=True   #permission
    )
    return vector_db.as_retriever(search_kwargs={"k": 3})

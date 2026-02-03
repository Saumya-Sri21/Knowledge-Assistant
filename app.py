import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from retrieval import get_retriever 

load_dotenv()

st.set_page_config(page_title="GenAI RAG Assistant", layout="wide")
st.title("📄 Knowledge Assistant")

retriever = get_retriever()
st.success("✅ documents loaded successfully!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

template = """
You are a specialized Knowledge Assistant.

Answer strictly from the provided context.

Rules:
- Use only the given context.
- If answer is not present, say:
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

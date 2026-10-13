import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Merchant of Venice RAG")
st.title("The Merchant of Venice - Q&A Bot")

@st.cache_resource
def initialize_rag_chain():
    url = "https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-merchant-of-venice_TXT_FolgerShakespeare.txt"
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant for Shakespeare's play 'The Merchant of Venice'.
Answer ONLY from the provided context. If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

with st.spinner("Initializing Vector Store..."):
    rag_chain = initialize_rag_chain()

question = st.text_input("Ask a question about The Merchant of Venice:")

if question:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(question)
        st.subheader("Answer:")
        st.write(response)
# The Merchant of Venice - RAG Q&A Bot

A local Retrieval-Augmented Generation (RAG) web application built using **Streamlit**, **LangChain**, **FAISS**, and **Groq API** to query and chat with Shakespeare's play *The Merchant of Venice*.

---

## Features
- **Smart Chunking & Embeddings**: Uses Hugging Face (`all-MiniLM-L6-v2`) and FAISS for fast local vector searching.
- **Lightning-Fast LLM**: Powered by Groq's high-speed inference engine.
- **Caching Mechanism**: Uses `@st.cache_resource` to load the vector store only once, ensuring smooth and instant responses.
- **Interactive UI**: Built with Streamlit for a clean web interface.

---

## Tech Stack
- **UI**: Streamlit
- **Orchestration**: LangChain (LCEL)
- **Vector Store**: FAISS
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Groq API (`openai/gpt-oss-20b`)

---

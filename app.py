from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore

# Page configuration
st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="centered")

st.title("📄 Document Q&A Bot")
st.caption("Ask questions and get answers grounded in your uploaded PDF document.")

# Ensure Google API key is configured
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing Google API key! Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Helper to extract clean text from LangChain message content
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        return "\n".join(text_parts)
    return str(content)

# Initialize LLM model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# Session state initialization
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""


def document_process(file_path):
    # 1. Document loading
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("Could not extract any text from this PDF file.")

    # 2. Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # 3. Embeddings & vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_db = InMemoryVectorStore.from_documents(documents=split_docs, embedding=embeddings)

    return vector_db


# Sidebar controls
with st.sidebar:
    st.header("Document Settings")
    if st.session_state.document_uploaded:
        st.success(f"Loaded: **{st.session_state.document_name}**")
        if st.button("📤 Upload New Document"):
            st.session_state.document_uploaded = False
            st.session_state.vector_db = None
            st.session_state.document_name = ""
            st.rerun()

# Document upload section
if not st.session_state.document_uploaded:
    uploaded_file = st.file_uploader(label="Select a PDF document", type=["pdf"])
    if uploaded_file is not None:
        save_path = "./uploaded_document.pdf"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("Processing document and generating embeddings..."):
            try:
                st.session_state.vector_db = document_process(save_path)
                st.session_state.document_uploaded = True
                st.session_state.document_name = uploaded_file.name
                st.success("Document processed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {e}")

# Q&A section
if st.session_state.document_uploaded and st.session_state.vector_db:
    query = st.text_input("Ask anything about your document:", placeholder="e.g., What are the key takeaways?")
    if query:
        with st.spinner("Searching and generating response..."):
            try:
                documents = st.session_state.vector_db.similarity_search(query=query, k=3)
                context = "\n\n".join(doc.page_content for doc in documents)

                prompt = f"""You are a helpful assistant. Provide an accurate and comprehensive answer based ONLY on the provided context. If the answer cannot be found in the context, say that you do not know.

Context:
{context}

Question:
{query}
"""
                answer = llm.invoke(prompt)
                clean_answer = extract_text(answer.content)
                st.markdown("### Answer")
                st.write(clean_answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

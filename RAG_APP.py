import streamlit as st
from dotenv import load_dotenv
import os
import time

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)

if not openai_api_key:
    st.error("OPENAI_API_KEY is missing. Please configure your secrets.")
    st.stop()

st.set_page_config(page_title="Dynamic RAG with OpenAI", layout="wide")
st.image("images.jpg")  # Make sure the image exists in your project directory
st.title("Dynamic RAG with OpenAI, FAISS, and Llama3-like Prompting")

# Initialize session state
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    # Save uploaded file to disk
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(file.name)
                    docs.extend(loader.load())

                # Split documents into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = splitter.split_documents(docs)

                # Create embeddings using HuggingFace
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
                st.success("Documents processed successfully!")
        else:
            st.warning("Please upload at least one document.")

# Main Chat Interface
st.header("Chat with your Documents")

# Initialize OpenAI Chat Model
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.

<context>
{context}
</context>

Question: {input}
""")

# Show previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector is not None:
        with st.chat_message("user"):
            st.markdown(prompt_input)

        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": prompt_input})
            end_time = time.process_time()

            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                st.info(f"Response time: {end_time - start_time:.2f} seconds")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response['answer']
            })
    else:
        st.warning("Please process your documents before asking questions.")

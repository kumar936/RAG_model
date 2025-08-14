import streamlit as st
from dotenv import load_dotenv
import os
from langchain.groq import ChatGrog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import CombineDocumentsChain  # Check if this is the correct class
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalChain  # Ensure this is the correct import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader  # Corrected from PyPpFLoader
from langchain.huggingface import HuggingFaceEmbeddings  # Corrected import
import time

import streamlit as st
import os
import glob
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ and OpenAI API keys 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up the Streamlit title
st.title("Open-Source Legal Assistant for Indian Law")

# Initialize Streamlit session state attributes if they do not exist
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
You are an expert legal assistant specializing in Indian law. Answer the following questions based on the provided legal context only.
Please ensure that your responses are accurate, precise, and reflect the relevant legal principles.

<context>
{context}
<context>

Questions:
{input}
"""
)

def get_pdf_files(directory):
    """Retrieve a list of PDF files in the specified directory."""
    return glob.glob(os.path.join(directory, '*.pdf'))

def vector_embedding():
    if st.session_state.vectors is None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./papers")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        
        # Create a vector store and save the vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

        # Save processed documents to a separate folder
        save_processed_documents(st.session_state.final_documents)

def save_processed_documents(documents):
    # Create a directory to save processed documents
    processed_dir = './processed_documents'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save each document as a text file
    for idx, doc in enumerate(documents):
        with open(os.path.join(processed_dir, f'document_{idx + 1}.txt'), 'w', encoding='utf-8') as f:
            f.write(doc.page_content)

# Load existing processed PDF files
processed_files = set(get_pdf_files('./papers'))

# Check for new PDF files in the laws directory
new_files = processed_files - st.session_state.processed_files

if new_files:
    vector_embedding()
    # Update the processed files state
    st.session_state.processed_files.update(new_files)
    st.write("New documents have been embedded and stored.")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
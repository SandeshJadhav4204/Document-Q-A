import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile

# Load environment variables
load_dotenv()

# Get the API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit title and sidebar
st.title("Gemma Model Document Q&A")
st.sidebar.title("Settings")

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'loader' not in st.session_state:
    st.session_state.loader = None

if 'docs' not in st.session_state:
    st.session_state.docs = None

if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = None

if 'final_documents' not in st.session_state:
    st.session_state.final_documents = None

if 'vectors' not in st.session_state:
    st.session_state.vectors = None

# Initialize the language model and prompt template
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
""")

def vector_embedding(file_path):
    try:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(file_path)  # Load the uploaded file
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        st.write("Document processing completed successfully.")
    except Exception as e:
        st.write(f"Error processing document: {e}")

def load_document():
    uploaded_file = st.sidebar.file_uploader("Upload your document", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        vector_embedding(tmp_file_path)

# Sidebar for uploading document
with st.sidebar.expander("Upload Document"):
    load_document()

# Main content for asking questions
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and st.session_state.vectors:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    try:
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander for document context
        with st.expander("Document Context"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No document context found.")
    except Exception as e:
        st.write(f"Error retrieving answer: {e}")

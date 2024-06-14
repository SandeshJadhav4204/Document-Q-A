from flask import Flask, request, render_template, redirect, url_for
import os
import tempfile
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

app = Flask(__name__)
load_dotenv()

# Get the API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize session state
session_state = {
    'embeddings': None,
    'loader': None,
    'docs': None,
    'text_splitter': None,
    'final_documents': None,
    'vectors': None,
    'current_file': None,
    'answer': None,
    'context': None,
    'message': None,
    'error': None,
    'response_time': None
}

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

def vector_embedding(file_path, filename):
    try:
        session_state['embeddings'] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        session_state['loader'] = PyPDFLoader(file_path)
        session_state['docs'] = session_state['loader'].load()
        session_state['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        session_state['final_documents'] = session_state['text_splitter'].split_documents(session_state['docs'])
        session_state['vectors'] = FAISS.from_documents(session_state['final_documents'], session_state['embeddings'])
        session_state['current_file'] = filename
        session_state['message'] = "Document processing completed successfully."
    except Exception as e:
        session_state['error'] = f"Error processing document: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                vector_embedding(tmp_file_path, uploaded_file.filename)
        if 'newFile' in request.files:
            new_uploaded_file = request.files['newFile']
            if new_uploaded_file.filename != '':
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(new_uploaded_file.read())
                    tmp_file_path = tmp_file.name
                vector_embedding(tmp_file_path, new_uploaded_file.filename)
        if 'question' in request.form:
            question = request.form['question']
            if question and session_state['vectors']:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = session_state['vectors'].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                try:
                    response = retrieval_chain.invoke({'input': question})
                    session_state['response_time'] = time.process_time() - start
                    session_state['answer'] = response.get('answer', 'No answer found.')
                    session_state['context'] = response.get('context', [])
                except Exception as e:
                    session_state['error'] = f"Error retrieving answer: {e}"
        return redirect(url_for('index'))
    return render_template('index.html', **session_state)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))


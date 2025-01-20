import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import gspread
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Google Sheets API setup
SERVICE_ACCOUNT_FILE = "service_account.json"  # Replace with your service account JSON file
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Streamlit page configuration
st.set_page_config(page_title="Multi-Document Question Answering System", layout="wide")
st.title("Multi-Document Question Answering System")
st.caption("Upload multiple Excel files or provide multiple Google Sheets links to ingest data into a vector store.")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Function to fetch data from Google Sheets
def fetch_google_sheet_data(sheet_url):
    """Fetches data from the given Google Sheets URL."""
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(credentials)
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)  # Fetch the first sheet
    data = worksheet.get_all_records()  # Returns data as a list of dictionaries
    return data

# Function to process data into text format
def process_data(data):
    """Converts data into text format for vector embeddings."""
    text_data = []
    for row in data:
        row_text = " | ".join([f"{key}: {value}" for key, value in row.items() if value])
        text_data.append(row_text)
    return text_data

# Vector embedding function
def vector_embedding(uploaded_files=None, sheet_urls=None):
    """Ingests data from multiple sources into the vector store."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        combined_data = []  # To store all processed data

        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                df = pd.read_excel(file)
                data = df.to_dict(orient="records")
                combined_data.extend(data)

        # Process Google Sheets
        if sheet_urls:
            for url in sheet_urls:
                data = fetch_google_sheet_data(url)
                combined_data.extend(data)

        # Convert combined data into text format and split into chunks
        text_data = process_data(combined_data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.create_documents(text_data)

        # Create vector store
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.write("Data ingestion complete. Vector store is ready.")

# User input for multiple documents
st.subheader("Upload Multiple Documents:")
uploaded_files = st.file_uploader("Upload Excel files (.xlsx):", type="xlsx", accept_multiple_files=True)

st.subheader("Provide Google Sheets Links:")
sheet_urls_input = st.text_area(
    "Enter Google Sheets URLs (one per line):",
    placeholder="https://docs.google.com/spreadsheets/d/...",
)

# User input for questions
prompt1 = st.text_input("Enter your question:")

# Ingest data into vector store
if st.button("Ingest the Data into Vector Store"):
    sheet_urls = [url.strip() for url in sheet_urls_input.split("\n") if url.strip()]
    if uploaded_files or sheet_urls:
        vector_embedding(uploaded_files=uploaded_files, sheet_urls=sheet_urls)
    else:
        st.error("Please upload files or provide Google Sheets links to ingest data.")

# Query the vector store
if prompt1:
    try:
        if "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write(f"Response time: {time.process_time() - start:.2f} seconds")
            st.write(response.get('answer', "No answer found."))
        else:
            st.warning("Please ingest data first by uploading files or providing Google Sheets links.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

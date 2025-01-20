RAG

### Document Question Answering System
This project is a Streamlit-based application that allows users to upload documents (Excel files or Google Sheets) and ask questions based on the ingested data. The application leverages LangChain, FAISS, and Google Generative AI for document ingestion, vector database creation, and generating answers to user queries.

### Features
Stage 1: Upload Excel Sheet
Users can upload a single .xlsx file.
The application processes the Excel sheet and creates a vector database using the content.
Users can query the content and receive precise answers based on the ingested data.

Stage 2: Added Google Drive Link Support
Users can provide a Google Sheets link instead of uploading an Excel file.
The application fetches data from the Google Sheet, processes it, and adds it to the vector database.
Supports integration with the Google Sheets API using a service account.

Stage 3: Multiple Document Chat
Users can upload multiple Excel files and/or provide multiple Google Sheets links simultaneously.
The application processes and combines data from all documents into a single vector database.
Users can query across all documents in one seamless search.


Technologies Used
Streamlit: User interface for uploading documents and querying data.
LangChain: Document processing, chain creation, and LLM integration.
FAISS: Vector database for efficient retrieval of document embeddings.
Google Sheets API: Integration for fetching data from Google Sheets.
Google Generative AI (Gemini): Language model for generating answers.

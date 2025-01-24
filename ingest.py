from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import time
# List of CSV file paths
DATA_PATHS = [
    "data/textbook_details_no_dup.csv",
    "data/textbook_details_with_disciplines.csv",
    "data/OpenTextbookLibrary.csv"
]
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    all_documents = []

    # Load and process each CSV file
    for data_path in DATA_PATHS:
        # Load documents from the CSV file
        loader = CSVLoader(file_path=data_path,encoding="utf-8")
        documents = loader.load()
        all_documents.extend(documents)  # Combine documents from all files
    # Optional: Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store from the combined documents and embeddings
    db = FAISS.from_documents(texts, embeddings)
    # Save the FAISS index locally
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

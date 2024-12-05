import os
import warnings
from typing import List
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return its documents.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        List[Document]: List of extracted documents
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (List[Document]): Input documents
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List[Document]: Split documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document]):
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): Input documents
    
    Returns:
        Chroma: Vector store
    """
    try:
        # Use Google's embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vector_store):
    """
    Create a question-answering chain.
    
    Args:
        vector_store: Vector store to use as retriever
    
    Returns:
        RetrievalQA: Question-answering chain
    """
    try:
        # Use Gemini 1.5 Pro model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", 
            temperature=0.1, 
            convert_system_message_to_human=True,
            # Optional: Add additional configuration for 1.5 Pro
            model_kwargs={
                "max_output_tokens": 8192,  # Leverage 1.5 Pro's large context window
                "top_k": 40,
                "top_p": 0.95
            }
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 5
            }
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

def main():
    # PDF file path
    pdf_path = "ril.pdf"
    
    # Load PDF
    documents = load_pdf(pdf_path)
    if not documents:
        return
    
    # Split documents
    split_docs = split_documents(documents)
    
    # Create vector store
    vector_store = create_vector_store(split_docs)
    if not vector_store:
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(vector_store)
    if not qa_chain:
        return
    
    # Example query
    query = "What is the annual revenue of the company?"
    
    try:
        # Run query
        response = qa_chain.invoke({"query": query})
        
        # Print response and source documents
        print("Answer:", response['result'])
        print("\nSource Documents:")
        for doc in response['source_documents']:
            print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:400]}...")
    
    except Exception as e:
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
import os
import warnings
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

class PDFChatbot:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.loaded_documents = []
        self.custom_system_prompt = "You are a helpful AI assistant specialized in analyzing PDF documents."

    def load_pdf(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDF files and return their documents."""
        all_documents = []
        try:
            for file_path in file_paths:
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            return all_documents
        except Exception as e:
            print(f"Error loading PDFs: {e}")
            return []

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> Optional[Chroma]:
        """Create a vector store from documents."""
        try:
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
            return self.vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def create_qa_chain(self, system_prompt: Optional[str] = None) -> Optional[RetrievalQA]:
        """Create a question-answering chain with optional custom system prompt."""
        if not self.vector_store:
            return None

        try:
            # Use custom system prompt if provided, otherwise use default
            effective_system_prompt = system_prompt or self.custom_system_prompt
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest", 
                temperature=0.1, 
                convert_system_message_to_human=True,
                system_prompt=effective_system_prompt,
                model_kwargs={
                    "max_output_tokens": 8192,  
                    "top_k": 10,
                    "top_p": 0.95
                }
            )
            
            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 10}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            return self.qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return None

    def process_pdf_query(self, query: str) -> str:
        """Process query against loaded PDFs."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please load PDFs first."

        try:
            response = self.qa_chain.invoke({"query": query})
            
            # Prepare top chunks
            top_chunks = "\n\n".join([
                f"Chunk (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:500]}"
                for doc in response['source_documents']
            ])

            return f"Top Relevant Chunks:\n{top_chunks}\n\nFinal Answer:\n{response['result']}"

        except Exception as e:
            return f"Error processing query: {e}"

    def initialize_pdfs(self, pdf_files, custom_system_prompt: Optional[str] = None):
        """Initialize vector store and QA chain from multiple PDFs."""
        # Reset loaded documents
        self.loaded_documents = []

        # Load PDFs
        documents = self.load_pdf(pdf_files)
        if not documents:
            return "Error: Could not load PDFs"

        # Split documents
        split_docs = self.split_documents(documents)

        # Store loaded documents
        self.loaded_documents = split_docs

        # Create vector store
        if not self.create_vector_store(split_docs):
            return "Error: Could not create vector store"

        # Create QA chain with optional custom system prompt
        if custom_system_prompt:
            self.custom_system_prompt = custom_system_prompt

        if not self.create_qa_chain(custom_system_prompt):
            return "Error: Could not create QA chain"

        # Return summary of loaded PDFs
        pdf_names = [os.path.basename(f) for f in pdf_files]
        return f"PDFs loaded and processed successfully!\nLoaded files: {', '.join(pdf_names)}\n" \
               f"Total document chunks: {len(split_docs)}"

def launch_gradio():
    """Launch Gradio interface for Multi-PDF Q&A."""
    chatbot = PDFChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Multi-PDF Q&A Chatbot")
        
        with gr.Row():
            with gr.Column():
                # PDF Upload (multiple)
                pdf_inputs = gr.File(
                    label="Upload PDFs", 
                    file_types=['.pdf'], 
                    file_count="multiple"
                )
                
                # Custom System Prompt
                system_prompt_input = gr.Textbox(
                    label="Custom System Prompt (Optional)", 
                    placeholder="Enter a custom instruction for the AI...",
                    lines=3
                )
                
                load_btn = gr.Button("Load PDFs")
                status_output = gr.Textbox(label="Status")
            
            with gr.Column():
                # Query section
                query_input = gr.Textbox(label="Ask a Question")
                submit_btn = gr.Button("Ask Question")
                output = gr.Textbox(label="Response", lines=10)
        
        # Load PDFs
        load_btn.click(
            fn=chatbot.initialize_pdfs, 
            inputs=[pdf_inputs, system_prompt_input], 
            outputs=[status_output]
        )
        
        # Ask Question
        submit_btn.click(
            fn=chatbot.process_pdf_query, 
            inputs=[query_input], 
            outputs=[output]
        )

    demo.launch()

def main():
    launch_gradio()

if __name__ == "__main__":
    main()
import os
import warnings
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

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

    def load_pdf(self, file_paths: List[str]) -> (List[Document], Optional[str]):
        """Load multiple PDF files and return their documents and an optional error message."""
        all_documents = []
        try:
            for file_path in file_paths:
                if not file_path: # Handle cases where a file path might be None or empty
                    continue
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())
            if not all_documents and file_paths: # If no documents were loaded but files were provided
                return [], "Error: No documents could be loaded from the provided PDF files. Please ensure they are valid."
            return all_documents, None
        except FileNotFoundError as fnf_error:
            return [], f"Error loading PDF: File not found - {fnf_error.filename}"
        except Exception as e:
            return [], f"Error loading PDFs: {e}"

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> (Optional[Chroma], Optional[str]):
        """Create a vector store from documents and return it with an optional error message."""
        if not documents:
            return None, "Error: No documents provided to create vector store."
        try:
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
            return self.vector_store, None
        except Exception as e:
            return None, f"Error creating vector store: {e}"

    def create_qa_chain(self, system_prompt: Optional[str] = None) -> (Optional[RetrievalQA], Optional[str]):
        """Create a question-answering chain with optional custom system prompt and return it with an optional error message."""
        if not self.vector_store:
            return None, "Error: Vector store not available for QA chain creation."

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
            self.qa_chain = qa_chain # Set self.qa_chain on success
            return self.qa_chain, None
        except Exception as e:
            return None, f"Error creating QA chain: {e}"

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

    def initialize_pdfs(self, pdf_files, custom_system_prompt: Optional[str] = None, status_output_ref=None, query_input_ref=None, submit_btn_ref=None):
        """Initialize vector store and QA chain from multiple PDFs, yielding status updates."""
        # Initial state: disable query/submit
        yield {
            status_output_ref: "Starting PDF processing...",
            query_input_ref: gr.Textbox(interactive=False, placeholder="Load PDFs to ask questions"),
            submit_btn_ref: gr.Button(interactive=False)
        }

        self.loaded_documents = []
        self.vector_store = None
        self.qa_chain = None

        if not pdf_files:
            yield {
                status_output_ref: "Error: No PDF files provided.",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return

        actual_pdf_files_with_obj = [(f.name, f) for f in pdf_files if f and hasattr(f, 'name')]
        if not actual_pdf_files_with_obj:
            yield {
                status_output_ref: "Error: No valid PDF files provided (filename missing or invalid object).",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return
        
        actual_pdf_filepaths = [name for name, _ in actual_pdf_files_with_obj]

        yield {
            status_output_ref: "Processing PDF files...",
            query_input_ref: gr.Textbox(interactive=False),
            submit_btn_ref: gr.Button(interactive=False)
        }

        # Load PDFs
        # Iterate through files to yield individual loading messages
        loaded_documents_count = 0
        for i, file_path in enumerate(actual_pdf_filepaths):
            yield {
                status_output_ref: f"Loading PDF {i+1}/{len(actual_pdf_filepaths)}: {os.path.basename(file_path)}...",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            # Note: self.load_pdf is designed to take a list. We are calling it with all files later.
            # This loop is for UI feedback only.
            # If self.load_pdf were to be called per file, error handling would need adjustment here.

        # Actual loading of all PDFs
        documents, error_msg = self.load_pdf(actual_pdf_filepaths)
        if error_msg:
            yield {
                status_output_ref: error_msg,
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return
        if not documents:
            yield {
                status_output_ref: "Error: Could not load PDFs for an unknown reason.",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return

        # Split documents
        yield {
            status_output_ref: "Splitting documents...",
            query_input_ref: gr.Textbox(interactive=False),
            submit_btn_ref: gr.Button(interactive=False)
        }
        split_docs = self.split_documents(documents)
        self.loaded_documents = split_docs

        # Create vector store
        yield {
            status_output_ref: "Creating vector store...",
            query_input_ref: gr.Textbox(interactive=False),
            submit_btn_ref: gr.Button(interactive=False)
        }
        _, error_msg = self.create_vector_store(split_docs)
        if error_msg:
            yield {
                status_output_ref: error_msg,
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return

        if custom_system_prompt:
            self.custom_system_prompt = custom_system_prompt
            yield {
                status_output_ref: "Custom system prompt updated. Creating QA chain...",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
        else:
            yield {
                status_output_ref: "Creating QA chain...",
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
        
        _, error_msg = self.create_qa_chain(self.custom_system_prompt)
        if error_msg:
            yield {
                status_output_ref: error_msg,
                query_input_ref: gr.Textbox(interactive=False),
                submit_btn_ref: gr.Button(interactive=False)
            }
            return

        pdf_names = [os.path.basename(f) for f in actual_pdf_filepaths]
        success_message = (f"PDFs loaded and processed successfully!\n"
                           f"Loaded files: {', '.join(pdf_names)}\n"
                           f"Total document chunks: {len(split_docs)}")
        yield {
            status_output_ref: success_message,
            query_input_ref: gr.Textbox(interactive=True, placeholder="Ask your question about the PDFs..."),
            submit_btn_ref: gr.Button(interactive=True)
        }

def launch_gradio():
    """Launch Gradio interface for Multi-PDF Q&A."""
    chatbot = PDFChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Multi-PDF Q&A Chatbot")
        
        with gr.Row():
            with gr.Column(scale=1):
                # PDF Upload (multiple)
                pdf_inputs = gr.File(
                    label="Upload PDFs", 
                    file_types=['.pdf'], 
                    file_count="multiple"
                )
                
                # Custom System Prompt
                system_prompt_input = gr.Textbox(
                    label="Custom System Prompt (Optional)", 
                    placeholder="Enter a custom instruction for the AI (e.g., 'Focus on financial data')...",
                    lines=3
                )
                
                load_btn = gr.Button("Load PDFs")
                status_output = gr.Textbox(label="Status", lines=3, interactive=False, placeholder="System status will appear here...") # Made interactive False
            
            with gr.Column(scale=2):
                # Query section
                query_input = gr.Textbox(
                    label="Ask a Question", 
                    interactive=False, # Initially disabled
                    placeholder="Load PDFs to ask questions"
                )
                submit_btn = gr.Button("Ask Question", interactive=False) # Initially disabled
                output = gr.Textbox(label="Response", lines=10, interactive=False, placeholder="AI's answer will appear here...") # Made interactive False
        
        # Load PDFs
        # Pass references to the Gradio components that initialize_pdfs will update
        load_btn.click(
            fn=lambda pdf_files, prompt: chatbot.initialize_pdfs(pdf_files, prompt, status_output, query_input, submit_btn),
            inputs=[pdf_inputs, system_prompt_input], 
            outputs=[status_output, query_input, submit_btn] # Target components for gr.update
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
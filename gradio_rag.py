
import os
import warnings
from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

warnings.filterwarnings("ignore")
load_dotenv()


class PDFChatbot:

    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.loaded_documents = []

        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Add it to your .env file."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.api_key
        )

        self.custom_system_prompt = """
You are an accurate PDF document analysis assistant.

Answer questions using only the information in the provided documents.

Rules:
1. Give complete and relevant answers.
2. Organize answers using headings and bullet points.
3. Do not invent facts or technologies.
4. Clearly distinguish explicitly mentioned information
   from information inferred from project descriptions.
5. If information is missing, say it is not available
   in the provided document.
6. Avoid unnecessary repetition.
7. Never claim that an inferred skill is explicitly listed.
"""

    def load_pdf(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDF files."""

        all_documents = []

        try:
            for file_path in file_paths:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)

            return all_documents

        except Exception as e:
            print(f"PDF loading error: {e}")
            return []

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> List[Document]:
        """Split documents into smaller chunks."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        return splitter.split_documents(documents)

    def create_vector_store(
        self,
        documents: List[Document]
    ) -> Optional[Chroma]:
        """Create ChromaDB vector store."""

        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            return self.vector_store

        except Exception as e:
            print(f"Vector store error: {e}")
            return None

    def create_qa_chain(
        self,
        system_prompt: Optional[str] = None
    ) -> Optional[RetrievalQA]:
        """Create Gemini-powered retrieval QA chain."""

        if self.vector_store is None:
            return None

        try:
            effective_prompt = (
                system_prompt.strip()
                if system_prompt and system_prompt.strip()
                else self.custom_system_prompt
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=self.api_key,
                temperature=0.1,
                max_output_tokens=2048,
                max_retries=1
            )

            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", effective_prompt),
                (
                    "human",
                    """Use the following retrieved document context
to answer the question.

Document context:
{context}

Question:
{question}

Provide a complete, accurate and well-organized answer.
Answer:"""
                )
            ])

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt
                }
            )

            return self.qa_chain

        except Exception as e:
            print(f"QA chain creation error: {e}")
            return None

    def initialize_pdfs(
        self,
        pdf_files,
        custom_system_prompt: Optional[str] = None
    ) -> str:
        """Load PDFs, split documents and initialize RAG."""

        try:
            if not pdf_files:
                return "Please upload at least one PDF."

            self.vector_store = None
            self.qa_chain = None
            self.loaded_documents = []

            documents = self.load_pdf(pdf_files)

            if not documents:
                return "Error: Could not load PDF documents."

            split_docs = self.split_documents(documents)

            if not split_docs:
                return "Error: No readable text found in PDFs."

            self.loaded_documents = split_docs

            if not self.create_vector_store(split_docs):
                return "Error: Could not create vector database."

            if not self.create_qa_chain(custom_system_prompt):
                return "Error: Could not initialize QA chain."

            pdf_names = [
                os.path.basename(path)
                for path in pdf_files
            ]

            return (
                "PDFs loaded successfully!\n\n"
                f"Files: {', '.join(pdf_names)}\n"
                f"Total pages: {len(documents)}\n"
                f"Total chunks: {len(split_docs)}\n"
                "Vector database: ChromaDB\n"
                "Embedding model: Gemini Embedding\n"
                "LLM: Gemini Flash Lite\n"
                "Status: Ready for questions."
            )

        except Exception as e:
            return f"PDF initialization error: {str(e)}"

    def process_pdf_query(self, query: str) -> str:
        """Retrieve relevant chunks and generate an answer."""

        if not self.qa_chain:
            return "Please upload and process PDFs first."

        if not query or not query.strip():
            return "Please enter a question."

        try:
            response = self.qa_chain.invoke({
                "query": query.strip()
            })

            answer = response.get("result", "")
            source_documents = response.get(
                "source_documents", []
            )

            if not answer:
                answer = "No answer was generated."

            source_text = []

            for index, doc in enumerate(source_documents, 1):

                page = doc.metadata.get("page", "N/A")

                source_name = os.path.basename(
                    doc.metadata.get("source", "Unknown")
                )

                source_text.append(
                    f"Source {index} | File: {source_name} "
                    f"| Page: {page}\n"
                    f"{doc.page_content[:500]}"
                )

            sources = "\n\n".join(source_text)

            return (
                f"FINAL ANSWER\n\n{answer}\n\n"
                f"{'=' * 50}\n"
                f"RETRIEVED SOURCE CHUNKS\n\n{sources}"
            )

        except Exception as e:
            return f"Query processing error: {str(e)}"


def launch_gradio():

    chatbot = PDFChatbot()

    with gr.Blocks(title="Multi-PDF RAG Assistant") as demo:

        gr.Markdown(
            """
            # Multi-PDF RAG Assistant

            Upload PDF documents and ask questions
            using Gemini-powered Retrieval-Augmented Generation.
            """
        )

        with gr.Row():

            with gr.Column():

                pdf_inputs = gr.File(
                    label="Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )

                system_prompt_input = gr.Textbox(
                    label="Custom System Prompt (Optional)",
                    placeholder="Enter additional AI instructions...",
                    lines=4
                )

                load_btn = gr.Button(
                    "Load and Process PDFs",
                    variant="primary"
                )

                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=6,
                    interactive=False
                )

            with gr.Column():

                query_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="Example: What technical skills does this person have?",
                    lines=2
                )

                submit_btn = gr.Button(
                    "Ask Question",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Answer and Retrieved Sources",
                    lines=22,
                    interactive=False
                )

        load_btn.click(
            fn=chatbot.initialize_pdfs,
            inputs=[
                pdf_inputs,
                system_prompt_input
            ],
            outputs=status_output
        )

        submit_btn.click(
            fn=chatbot.process_pdf_query,
            inputs=query_input,
            outputs=output
        )

        query_input.submit(
            fn=chatbot.process_pdf_query,
            inputs=query_input,
            outputs=output
        )

    demo.launch()


if __name__ == "__main__":
    launch_gradio()

import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import sys

# Add the parent directory to the sys.path to allow imports from rag.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from rag.py and langchain
import rag 
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Used by rag.py

# Check if GOOGLE_API_KEY is set, as rag.py raises an error if it's not on import.
# For testing, we mock all external calls, so a real key isn't used by the tests themselves.
# However, the module-level check in rag.py will execute.
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = 'test_key_for_rag_import'

class TestRag(unittest.TestCase):

    @patch('rag.PyPDFLoader')
    def test_load_pdf_success(self, MockPyPDFLoader):
        """Test successful PDF loading."""
        mock_loader_instance = MockPyPDFLoader.return_value
        mock_docs = [Document(page_content="doc1 content"), Document(page_content="doc2 content")]
        mock_loader_instance.load.return_value = mock_docs
        
        loaded_docs = rag.load_pdf("dummy.pdf")
        
        MockPyPDFLoader.assert_called_once_with("dummy.pdf")
        mock_loader_instance.load.assert_called_once()
        self.assertEqual(loaded_docs, mock_docs)

    @patch('rag.PyPDFLoader')
    @patch('builtins.print') # To capture print statements
    def test_load_pdf_file_not_found(self, mock_print, MockPyPDFLoader):
        """Test PDF loading when file is not found."""
        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.side_effect = FileNotFoundError("File not found: missing.pdf")
        
        loaded_docs = rag.load_pdf("missing.pdf")
        
        MockPyPDFLoader.assert_called_once_with("missing.pdf")
        mock_loader_instance.load.assert_called_once()
        self.assertEqual(loaded_docs, [])
        mock_print.assert_any_call("Error: File missing.pdf not found.")

    @patch('rag.PyPDFLoader')
    @patch('builtins.print')
    def test_load_pdf_other_exception(self, mock_print, MockPyPDFLoader):
        """Test PDF loading with a generic exception."""
        mock_loader_instance = MockPyPDFLoader.return_value
        mock_loader_instance.load.side_effect = Exception("Some other error")

        loaded_docs = rag.load_pdf("any.pdf")

        MockPyPDFLoader.assert_called_once_with("any.pdf")
        mock_loader_instance.load.assert_called_once()
        self.assertEqual(loaded_docs, [])
        mock_print.assert_any_call("Error loading PDF: Some other error")


    def test_split_documents(self):
        """Test document splitting functionality."""
        docs = [
            Document(page_content="This is a long document that needs splitting. " * 50),
            Document(page_content="Another long document for splitting. " * 50)
        ]
        # Using parameters from rag.py's split_documents for consistency
        # chunk_size=1000, chunk_overlap=200
        # Default chunk_size for RecursiveCharacterTextSplitter is 4000 if not specified.
        # rag.py specifies chunk_size=1000
        
        split_docs = rag.split_documents(docs, chunk_size=100, chunk_overlap=20) 
        
        self.assertIsInstance(split_docs, list)
        self.assertTrue(all(isinstance(doc, Document) for doc in split_docs))
        
        # Expect more documents than original if splitting occurred
        self.assertTrue(len(split_docs) > len(docs)) 
        
        for doc in split_docs:
            # Check if content is shorter than or equal to chunk_size
            # This is a basic check; actual length can vary slightly due to splitter logic
            self.assertTrue(len(doc.page_content) <= 100 + 50) # Allow for some overlap and boundary conditions

    @patch('rag.GoogleGenerativeAIEmbeddings')
    @patch('rag.Chroma')
    def test_create_vector_store_success(self, MockChroma, MockGoogleEmbeddings):
        """Test successful vector store creation."""
        mock_embeddings_instance = MockGoogleEmbeddings.return_value
        mock_chroma_instance = MockChroma.from_documents.return_value
        
        docs = [Document(page_content="doc content")]
        vector_store = rag.create_vector_store(docs)
        
        MockGoogleEmbeddings.assert_called_once_with(model="models/embedding-001")
        MockChroma.from_documents.assert_called_once_with(docs, mock_embeddings_instance)
        self.assertEqual(vector_store, mock_chroma_instance)

    @patch('rag.GoogleGenerativeAIEmbeddings')
    @patch('rag.Chroma')
    @patch('builtins.print')
    def test_create_vector_store_failure(self, mock_print, MockChroma, MockGoogleEmbeddings):
        """Test vector store creation failure."""
        MockGoogleEmbeddings.return_value # Standard mock instance
        MockChroma.from_documents.side_effect = Exception("Chroma error")
        
        docs = [Document(page_content="doc content")]
        vector_store = rag.create_vector_store(docs)
        
        self.assertIsNone(vector_store)
        mock_print.assert_any_call("Error creating vector store: Chroma error")

    @patch('rag.RetrievalQA')
    @patch('rag.ChatGoogleGenerativeAI')
    def test_create_qa_chain_success(self, MockChatGoogleAI, MockRetrievalQA):
        """Test successful QA chain creation."""
        mock_llm_instance = MockChatGoogleAI.return_value
        mock_qa_chain_instance = MockRetrievalQA.from_chain_type.return_value
        
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        qa_chain = rag.create_qa_chain(mock_vector_store)
        
        MockChatGoogleAI.assert_called_once_with(
            model="gemini-1.5-pro-latest", 
            temperature=0.1, 
            convert_system_message_to_human=True,
            model_kwargs={
                "max_output_tokens": 8192,  
                "top_k": 10,
                "top_p": 0.95
            }
        )
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        MockRetrievalQA.from_chain_type.assert_called_once_with(
            llm=mock_llm_instance, 
            chain_type="stuff", 
            retriever=mock_retriever,
            return_source_documents=True
        )
        self.assertEqual(qa_chain, mock_qa_chain_instance)

    @patch('rag.RetrievalQA')
    @patch('rag.ChatGoogleGenerativeAI')
    @patch('builtins.print')
    def test_create_qa_chain_failure(self, mock_print, MockChatGoogleAI, MockRetrievalQA):
        """Test QA chain creation failure."""
        MockChatGoogleAI.return_value # Standard mock instance
        MockRetrievalQA.from_chain_type.side_effect = Exception("RetrievalQA error")
        
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock() # Standard retriever mock
        
        qa_chain = rag.create_qa_chain(mock_vector_store)
        
        self.assertIsNone(qa_chain)
        mock_print.assert_any_call("Error creating QA chain: RetrievalQA error")

if __name__ == '__main__':
    unittest.main()

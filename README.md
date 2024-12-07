# Multi PDF Chatbot with Gradio UI using Langchain and Google Gemini 

## Overview
This application enables question-answering over PDF documents using Google's Gemini AI and LangChain, with vector-based document retrieval.

## Prerequisites
- Python 3.8+
- Google AI API Key

## Dependencies
```bash
pip install -r requirements.txt 
```

## Environment Setup
1. Create a `.env` file
2. Add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Configuration
- Modify `pdf_path` in `main()` to specify your PDF
- Adjust chunk size/overlap in `split_documents()`
- Customize LLM parameters in `create_qa_chain()`

## Features
- PDF document loading
- Recursive text splitting
- Vector embedding with ChromaDB
- Retrieval-augmented question answering
- Source document tracking

## Usage
- For Basic RAG
```python
python rag.py
```
- For Multi PDF RAG with Gradio
```python
python gradio_rag.py
```


## Customization
- Change embedding/language models
- Modify retrieval search parameters
- Adjust temperature and output settings


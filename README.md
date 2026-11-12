# Gemini RAG – Multi-PDF Question Answering Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions using Google's Gemini models.

## Features

- Upload and process multiple PDF documents.
- Extract text from PDF files.
- Split documents into smaller chunks.
- Generate embeddings using Gemini Embedding.
- Store document vectors in ChromaDB.
- Retrieve relevant document chunks using similarity search.
- Generate contextual answers using Gemini.
- Support custom system prompts.
- Display retrieved source chunks alongside answers.
- Interactive Gradio interface.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Frontend | Gradio |
| Environment | Python-dotenv |

## Project Structure

```text
gemini-rag/
├── gradio_rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd gemini-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

Get an API key from Google AI Studio.

### 5. Run the application

```bash
python gradio_rag.py
```

Open the local Gradio URL displayed in your terminal.

## How It Works

1. Upload PDF documents.
2. Extract document text using PyPDFLoader.
3. Split text into smaller chunks.
4. Generate embeddings using Gemini.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks for user questions.
7. Pass retrieved context to Gemini.
8. Display the generated answer and source chunks.

## Future Improvements

- Conversation history
- Streaming responses
- Improved source citations
- RAG evaluation
- Retrieval optimization
- Deployment

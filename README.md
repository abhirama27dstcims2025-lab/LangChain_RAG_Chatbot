# LangChain RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, Hugging Face embeddings, FAISS, and Google Gemini. The chatbot retrieves relevant information from a text document and uses an LLM to generate context-based answers to user questions.

## Project Overview

This project demonstrates how to build a basic document-based question-answering system using RAG.

Instead of relying only on the language model's existing knowledge, the chatbot retrieves relevant text chunks from a local document and supplies them as context when generating an answer.

## Features

- **Document Loading:** Loads text documents using LangChain's `TextLoader`.
- **Text Splitting:** Divides documents into smaller chunks using `RecursiveCharacterTextSplitter`.
- **Text Embeddings:** Converts text chunks into numerical vectors using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model.
- **Vector Search:** Uses FAISS to store and search document embeddings.
- **Context Retrieval:** Retrieves the three most relevant chunks for a user question.
- **AI-Generated Answers:** Uses Google Gemini to generate answers based on the retrieved context.
- **Interactive Chat:** Allows users to ask multiple questions through the terminal.
- **Source Information:** Displays metadata for the retrieved document chunks.

## Technologies Used

- Python
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Google Gemini API
- python-dotenv

## Project Structure

```text
Rag_Project/
├── documents/
│   └── sample_profile.txt
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

**Note:** The `.env` file is used locally to store the Gemini API key. It must not be uploaded to GitHub.

## How It Works

The chatbot follows this RAG workflow:

1. Load a text document.
2. Split the document into smaller text chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in a FAISS vector store.
5. Retrieve relevant chunks based on the user's question.
6. Pass the retrieved context and question to Google Gemini.
7. Display the generated answer and source metadata.
## Example Questions

After the chatbot starts, try asking:

- What is my name?
- What course am I currently pursuing?
- What is my educational background?
- What technical skills are mentioned in the document?
- Summarize the information in the document.

The answers depend on the content of the document you provide.

To stop the chatbot, type:

```text
exit
```

## Configuration

The current implementation uses the following settings:

| Component | Configuration |
|---|---|
| Text chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Retrieved chunks | 3 |
| Language model | Google Gemini |
| Temperature | 0 |

These settings can be adjusted in `main.py`.

## Limitations

- Currently supports text documents through `TextLoader`.
- The FAISS index is recreated each time the application runs.
- Conversation history is not retained between questions.
- Answer quality depends on the source document and retrieved context.
- A valid Gemini API key and internet access are required for answer generation.

## Future Improvements

- Add PDF document support.
- Persist and reload the FAISS vector index.
- Add conversational memory for follow-up questions.
- Build a web interface using Streamlit.
- Support multiple documents and document sources.



---

*This project was developed as a learning project to understand document retrieval, vector embeddings, and Retrieval-Augmented Generation using LangChain.*

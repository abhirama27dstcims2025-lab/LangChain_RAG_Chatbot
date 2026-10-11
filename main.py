
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# 1. Load the text document
loader = TextLoader(
    "documents/abhijit_profile.txt",
    encoding="utf-8"
)
documents = loader.load()

print("Document loaded successfully!")

# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)
print("Number of chunks:", len(chunks))

# 3. Convert chunks into embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

print("FAISS vector store created!")

# 5. Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 6. Configure Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

# 7. Create the prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the supplied context. "
        "If the answer is not present in the context, "
        "say you could not find it in the document.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

# 8. Ask questions
while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower().strip() == "exit":
        break

    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    # Generate an answer
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    print("\nAnswer:", response.text)

    # Display source information
    print("\nSource documents:")
    for doc in relevant_docs:
        print(doc.metadata)



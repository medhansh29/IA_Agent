import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document # Import Document for type hinting if needed
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- Step 1: Load and Prepare Data ---
def load_csv_data(file_path):
    """Loads CSV data using pandas."""
    try:
        df = pd.read_csv(file_path)
        documents = []
        for index, row in df.iterrows():
            # Customize this based on what information from each row is relevant.
            # Convert NaN values to empty strings or a sensible default
            row_data = {k: v if pd.notna(v) else 'N/A' for k, v in row.to_dict().items()}
            
            doc_content = (
                f"Flow ID: {row_data.get('flow_id', 'N/A')}, "
                f"Category: {row_data.get('category', 'N/A')}, "
                f"Subcategory: {row_data.get('subcategory', 'N/A')}, "
                f"Description: {row_data.get('description', 'N/A')}, "
                f"Tags: {row_data.get('tags', 'N/A')}"
            )
            # You can also add metadata if your vectorstore supports it and you need it
            # documents.append(Document(page_content=doc_content, metadata={"source": file_path, "row_index": index}))
            documents.append(doc_content)
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred loading the CSV: {e}")
        return []

# --- Step 2: Create Embeddings and Vector Store ---
def setup_vector_store(documents, persist_directory="./chroma_db"):
    """
    Creates embeddings and stores them in a ChromaDB vector store.
    The vector store will be persisted to disk.
    """
    embeddings = OpenAIEmbeddings()
    
    # Check if the persist directory exists and contains data
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"Creating new vector store and ingesting {len(documents)} documents...")
        # Chroma.from_texts creates and ingests documents
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print("Vector store created and persisted.")
    return vectorstore

# --- Step 3: Set up RAG Chain ---
def setup_rag_chain(vectorstore):
    """Sets up the Retrieval Augmented Generation chain for standalone testing."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # Using a compact model
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    template = """You are an AI assistant helping a fintech company. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    rag_prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- NEW: Function to get RAG chain and retriever for external use ---
def get_rag_chain_and_retriever():
    """
    Initializes and returns both the RAG chain (for direct querying)
    and the retriever (for fetching documents) from the vector store.
    """
    csv_file_path = "income_assesment/data_store/filtered_flow_data.csv"
    documents = load_csv_data(csv_file_path)
    if not documents:
        raise ValueError(f"Could not load documents from {csv_file_path}. Ensure file exists and is accessible.")
    
    vectorstore = setup_vector_store(documents)
    
    # The retriever for fetching relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # The full RAG chain (useful for standalone testing)
    rag_chain_for_testing = setup_rag_chain(vectorstore) 
    
    return rag_chain_for_testing, retriever

# --- Main Execution Flow for Standalone Testing ---
if __name__ == "__main__":
    print("Running standalone RAG demo...")
    try:
        rag_chain, _ = get_rag_chain_and_retriever() # Get only the rag_chain for testing
        print("\n--- RAG Model Ready (Standalone Demo) ---")
        print("Enter your questions (type 'exit' to quit):")

        while True:
            user_query = input("\nYour Question: ")
            if user_query.lower() == 'exit':
                print("Exiting RAG demo.")
                break
            
            try:
                response = rag_chain.invoke(user_query)
                print(f"RAG Response: {response}")
            except Exception as e:
                print(f"An error occurred during RAG query: {e}")
    except Exception as e:
        print(f"Failed to initialize RAG demo: {e}")
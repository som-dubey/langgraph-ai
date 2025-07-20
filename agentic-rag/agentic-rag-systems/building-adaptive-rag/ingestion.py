# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
# from model import embed_model

# load_dotenv()

# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=250, chunk_overlap=0
# )

# doc_splits = text_splitter.split_documents(docs_list)

# embed = embed_model

# # Create vector store with documents
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embed,
#     persist_directory="./.chroma",
# )

# # Create retriever
# retriever = vectorstore.as_retriever()

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from model import embed_model

load_dotenv()

def create_vectorstore():
    """Create vector store only if it doesn't exist"""
    
    # Check if vector store already exists
    chroma_path = "./.chroma"
    if os.path.exists(chroma_path) and os.listdir(chroma_path):
        print("📚 Loading existing vector store...")
        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=embed_model,
            persist_directory=chroma_path,
        )
        return vectorstore.as_retriever()
    
    print("🔄 Creating new vector store...")
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    # Create vector store with documents
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embed_model,
        persist_directory=chroma_path,
    )
    
    print("✅ Vector store created and persisted!")
    return vectorstore.as_retriever()

# Initialize retriever (will create vectorstore if needed)
retriever = create_vectorstore()
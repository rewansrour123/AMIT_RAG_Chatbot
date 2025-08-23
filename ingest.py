
# TODO: Implement logic to ingest documents from the docs/ folder
# TODO: Generate embeddings for each document using the embedding service
# TODO: Store the embeddings in the vector database
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings   

# 1) Load all text files from docs/
docs = []
for file in os.listdir("docs"):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join("docs", file))
        docs.extend(loader.load())

# 2) Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# 3) Create embeddings
embeddings = FakeEmbeddings(size=128)

# 4) Store them in Chroma vector database
db = Chroma.from_documents(documents, embeddings, persist_directory="vectorstore")

print("✅ Ingestion completed! Data stored in vectorstore/")


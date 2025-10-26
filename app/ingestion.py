from langchain.text_splitters import CharacterTextSplitter
from langchain_community.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
import numpy as np
import faiss
import os

def ingest_docs(doc_folder):
    # Load and split documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for filename in os.listdir(doc_folder):
        with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            texts.extend(splitter.split_text(text))
    # Create embeddings
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = embedder.embed_documents(texts)
    
    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    
    # Save index and texts for retrieval
    faiss.write_index(index, 'processed/faiss.index')
    with open('processed/texts.txt', 'w', encoding='utf-8') as f:
        f.write('\n|||\n'.join(texts))
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_docs('data/raw')

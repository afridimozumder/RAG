from langchain_community.embeddings import OllamaEmbeddings
import faiss
import numpy as np

def load_index():
    return faiss.read_index('processed/faiss.index')

def retrieve(query, index, top_k=5):
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    q_embedding = np.array([embedder.embed_query(query)])
    distances, indices = index.search(q_embedding, top_k)
    return indices[0]

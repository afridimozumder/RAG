import ollama
import numpy as np

from retriever import load_index, retrieve

def generate_answer(query, use_retrieval=True):
    prompt = ""
    if use_retrieval:
        # Load documents
        index = load_index()
        with open('processed/texts.txt', 'r', encoding='utf-8') as f:
            docs = f.read().split('\n|||\n')
        # Retrieve relevant docs using vector search
        indices = retrieve(query, index)
        context = "\n\n".join(docs[i] for i in indices)
        prompt = f"Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    else:
        # Just use the plain query without context
        prompt = f"Answer the following question as accurately as possible:\n\nQuestion: {query}\nAnswer:"

    # Call Ollama for generation
    response = ollama.chat(
        model="llama3",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

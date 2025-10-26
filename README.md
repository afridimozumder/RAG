# Distributed Retrieval-Augmented Generation (RAG) to mitigate hallucination in LLM

## Overview
This project implements a distributed Retrieval-Augmented Generation (RAG) system to reduce hallucinations in large language models (LLMs). It integrates:

- Ollama local LLMs for text generation and embeddings
- FAISS for vector similarity search
- LangChain for text splitting
- Gradio UI for interactive querying with toggleable retrieval

A lightweight Ollama model (`phi`) is used to ensure smooth operation on 8 GB RAM laptops.

***

## Features

- Ingests your own text documents to build searchable vector stores
- Retrieves relevant context during query time to augment LLM answers
- Toggleable retrieval for direct comparison of hallucination reduction
- Simple web UI through Gradio for easy experimentation

***

## Setup Instructions

### 1. Create Python Virtual Environment and Activate

```bash
python -m venv rag_env
rag_env\Scripts\activate   # Windows
# source rag_env/bin/activate  # Linux/Mac
```

### 2. Install Required Packages

```bash
pip install numpy faiss-cpu langchain langchain-community gradio ollama
```

### 3. Install Ollama and Pull Models

- Download and install Ollama from https://ollama.com
- In terminal, pull models:

```bash
ollama pull phi
ollama pull nomic-embed-text:latest
```

### 4. Prepare Ingestion Data

Place text files in `data/raw/`

Run ingestion to build the index:

```bash
python ingestion.py
```

### 5. Launch Gradio UI

```bash
python gradio_ui.py
```

Open browser at [http://127.0.0.1:7860](http://127.0.0.1:7860)  
Type queries and toggle retrieval on/off to test hallucination mitigation.

***

## File Structure and Functionality

- **ingestion.py**: Splits and embeds text documents, builds FAISS vector index
- **retriever.py**: Searches vector store to find top relevant documents
- **llm.py**: Calls Ollama LLM with or without retrieved context for answer generation
- **gradio_ui.py**: Web UI frontend with toggling of context retrieval

## Future Work

- Scale to more distributed peers for improved data privacy.
- Integrate advanced peer discovery algorithms like TARW.
- Add persistent chat history and richer UI features.

***

This project is ideal for research and experimentation with mitigating hallucinations in LLMs using distributed retrieval-augmented generation.

***

[1](https://github.com/Tublian/langchain-rag-template)
[2](https://www.reddit.com/r/LocalLLaMA/comments/1e5n96c/lemme_see_your_best_rag_projects/)
[3](https://redis.io/blog/announcing-langchain-rag-template-powered-by-redis/)
[4](https://www.youtube.com/watch?v=Z8z1ae0-SFg)
[5](https://www.reddit.com/r/AI_Agents/comments/1iix4k8/i_built_an_ai_agent_that_creates_readme_file_for/)
[6](https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e)
[7](https://packaging.python.org/guides/making-a-pypi-friendly-readme/)
[8](https://www.makeareadme.com)
[9](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/blob/main/README.md)
[10](https://developer.ibm.com/tutorials/build-rag-assistant-md-documentation/)

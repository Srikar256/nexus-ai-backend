# Nexus AI: Semantic Game Discovery Engine

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) 
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone)
![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)

Nexus AI is a full-stack, AI-powered search engine that enables users to discover video games based on concepts, lore, and gameplay mechanics using natural language processing rather than exact keyword matching.

## 🧠 System Architecture

*   **Semantic Pipeline:** Cleans unstructured dataset metadata and serializes it into natural language narratives for high-context embedding generation.
*   **Vector Engine:** Utilizes approximate nearest neighbor (ANN) search to match user queries with game vectors based on cosine similarity.
*   **Serverless Offloading:** Bypasses low-memory container bottlenecks by delegating heavy AI math to external cloud clusters, reducing local API memory overhead by over 80%.

## ⚙️ Tech Stack

*   **Frontend Interface:** Built with React and Vite for a lightning-fast, edge-deployed UI.
*   **Backend Infrastructure:** Asynchronous FastAPI server deployed on Render.
*   **Machine Learning:** Powered by the BAAI/bge-small-en-v1.5 dense vector model via the Hugging Face Inference API.
*   **Database:** Pinecone vector database for sub-second, high-dimensional similarity scoring.

## 🚀 Quick Start

1.  Clone this repository and navigate to the root directory.
2.  Create a .env file and add your PINECONE_API_KEY and HF_TOKEN.
3.  Install the dependencies listed in requirements.txt.
4.  Boot the Uvicorn server to initialize the API endpoints.
5.  In a separate terminal, install the frontend modules and launch the Vite development server.

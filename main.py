import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from huggingface_hub import InferenceClient


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set")


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(title="Nexus AI Semantic Search")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Pinecone
# --------------------------------------------------

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("semantic-search")

print("Connected to Pinecone successfully.")


# --------------------------------------------------
# Hugging Face
# --------------------------------------------------

print("Connecting to Hugging Face...")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

HF_MODEL = "BAAI/bge-small-en-v1.5"

print(f"Hugging Face model configured: {HF_MODEL}")


# --------------------------------------------------
# Search endpoint
# --------------------------------------------------

@app.get("/search")
def search_database(q: str):

    if not q or not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    try:

        # BGE query instruction
        formatted_query = (
            "Represent this sentence for searching relevant passages: "
            + q.strip()
        )

        print(f"Generating embedding for: {q}")

        # Generate embedding using Hugging Face
        embedding = hf_client.feature_extraction(
            formatted_query,
            model=HF_MODEL,
        )

        # Convert numpy array to Python list
        query_vector = embedding.tolist()

        # Hugging Face may return [[...384 values...]]
        # Pinecone needs [...]
        if (
            isinstance(query_vector, list)
            and len(query_vector) > 0
            and isinstance(query_vector[0], list)
        ):
            query_vector = query_vector[0]

        print(f"Embedding generated. Dimensions: {len(query_vector)}")

    except Exception as e:

        print(f"Hugging Face embedding error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate search embedding."
        )


    # --------------------------------------------------
    # Pinecone search
    # --------------------------------------------------

    try:

        result = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
        )

    except Exception as e:

        print(f"Pinecone search error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Failed to search Pinecone."
        )


    # --------------------------------------------------
    # Format results
    # --------------------------------------------------

    matches = []

    for match in result.get("matches", []):

        metadata = match.get("metadata", {})

        matches.append(
            {
                "title": metadata.get("title", "Unknown"),
                "description": metadata.get("description", "Unknown"),
                "genre": metadata.get("genre", "Unknown"),
                "year": metadata.get("year", "Unknown"),
                "rating": metadata.get("rating", "Unknown"),
                "similarity_score": round(
                    match.get("score", 0),
                    4
                ),
            }
        )


    return {
        "search_query": q,
        "top_matches": matches,
    }
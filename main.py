import os
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI(title="Nexus AI Semantic Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Connecting to database...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("semantic-search")

# The free HuggingFace API Endpoint for your exact model
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en-v1.5"

@app.get("/search")
async def search_database(q: str):
    formatted_query = f"Represent this sentence for searching relevant passages: {q}"
    
    # 1. Ping HuggingFace to do the AI math (Uses 0MB of Render's RAM!)
    response = requests.post(HF_API_URL, json={"inputs": formatted_query})
    
    # Handle HuggingFace "Cold Start" (If their model is asleep, wait and retry)
    if response.status_code == 503:
        print("Model is asleep. Waiting 15 seconds for HuggingFace to wake it up...")
        time.sleep(15) 
        response = requests.post(HF_API_URL, json={"inputs": formatted_query}) 
        
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="HuggingFace API Error")
        
    query_vector = response.json()
    
    # Safely flatten the array just in case HuggingFace returns a nested list
    while isinstance(query_vector, list) and isinstance(query_vector[0], list):
        query_vector = query_vector[0]
        
    # 2. Search Pinecone
    result = index.query(
        vector=query_vector,
        top_k=5, 
        include_metadata=True
    )
    
    matches = []
    for match in result['matches']:
        matches.append({
            "title": match['metadata']['title'],
            "description": match['metadata']['description'],
            "genre": match['metadata']['genre'],
            "year": match['metadata']['year'],
            "rating": match['metadata']['rating'],
            "similarity_score": round(match['score'], 4)
        })
        
    return {"search_query": q, "top_matches": matches}
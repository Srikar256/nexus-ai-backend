import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
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

print("Waking up the BAAI model...")
# --- UPGRADED MODEL HERE ---
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

@app.get("/search")
async def search_database(q: str):
    # Add a prompt instruction that makes BAAI models perform even better
    formatted_query = f"Represent this sentence for searching relevant passages: {q}"
    query_vector = model.encode(formatted_query).tolist()
    
    result = index.query(
        vector=query_vector,
        top_k=5, # Pulling the top 5 best matches
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
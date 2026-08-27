import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("semantic-search")

print("Loading the State-of-the-Art BAAI Model...")
# --- UPGRADED MODEL HERE ---
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

print("Loading CSV dataset...")
df = pd.read_csv("imdb_video_games.csv")

# Clean the data
df = df.dropna(subset=['Summary'])
df = df.fillna('Unknown') 
df = df.head(2000)

batch_size = 100
total_batches = (len(df) // batch_size) + (1 if len(df) % batch_size != 0 else 0)

print(f"Starting batch upload in {total_batches} chunks...")

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    vectors_to_upload = []
    
    for idx, row in batch.iterrows():
        # Natural Language Formatting
        year_clean = str(row['Year']).replace('.0', '')
        rich_text = f"{row['Title']} is a {year_clean} {row['Genre']} game. It stars {row['Stars']}. Summary: {row['Summary']}"
        
        embedding = model.encode(rich_text).tolist()
        
        metadata = {
            "title": str(row['Title']),
            "description": str(row['Summary']),
            "genre": str(row['Genre']),
            "year": year_clean,
            "rating": str(row['User Rating'])
        }
        
        vectors_to_upload.append((str(idx), embedding, metadata))
    
    index.upsert(vectors=vectors_to_upload)
    print(f"Uploaded batch {(i//batch_size) + 1}/{total_batches}")

print("Success! Hyper-intelligent BAAI dataset injected into Pinecone.")
import os

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_NAME = "semantic-search"
CSV_FILE = "imdb_video_games.csv"

MAX_ROWS = 2000
BATCH_SIZE = 100


# --------------------------------------------------
# Connect to Pinecone
# --------------------------------------------------

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone successfully.")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print(f"Loading dataset: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

print(f"Original dataset rows: {len(df)}")


# --------------------------------------------------
# Clean dataset
# --------------------------------------------------

df = df.dropna(subset=["Summary"])

df = df.fillna("Unknown")

df = df.head(MAX_ROWS)

print(f"Rows selected for indexing: {len(df)}")


# --------------------------------------------------
# Create and upload embeddings
# --------------------------------------------------

total_batches = len(df) // BATCH_SIZE + (1 if len(df) % BATCH_SIZE != 0 else 0)

print(f"Starting upload in {total_batches} batches...")


for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i : i + BATCH_SIZE]

    texts = []
    metadata_list = []
    ids = []

    # ----------------------------------------------
    # Prepare text
    # ----------------------------------------------

    for idx, row in batch.iterrows():
        year_clean = str(row["Year"]).replace(".0", "")

        rich_text = (
            f"{row['Title']} is a "
            f"{year_clean} "
            f"{row['Genre']} game. "
            f"It stars {row['Stars']}. "
            f"Summary: {row['Summary']}"
        )

        texts.append(rich_text)

        ids.append(str(idx))

        metadata_list.append(
            {
                "title": str(row["Title"]),
                "description": str(row["Summary"]),
                "genre": str(row["Genre"]),
                "year": year_clean,
                "rating": str(row["User Rating"]),
            }
        )

    # ----------------------------------------------
    # Generate embeddings for entire batch
    # ----------------------------------------------

    print(f"Generating embeddings for batch {(i // BATCH_SIZE) + 1}/{total_batches}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    # ----------------------------------------------
    # Prepare Pinecone vectors
    # ----------------------------------------------

    vectors_to_upload = []

    for vector_id, embedding, metadata in zip(
        ids,
        embeddings,
        metadata_list,
    ):
        vectors_to_upload.append(
            (
                vector_id,
                embedding.tolist(),
                metadata,
            )
        )

    # ----------------------------------------------
    # Upload to Pinecone
    # ----------------------------------------------

    index.upsert(vectors=vectors_to_upload)

    print(f"Uploaded batch {(i // BATCH_SIZE) + 1}/{total_batches}")


print()
print("==========================================")
print("SUCCESS!")
print("BGE embeddings uploaded to Pinecone.")
print("==========================================")

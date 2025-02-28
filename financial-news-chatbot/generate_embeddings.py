import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 🔹 Load financial news data
df_news = pd.read_csv("financial_news.csv")  # Load saved news articles

# 🔹 Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Generate embeddings for each news article
news_texts = df_news["title"] + " " + df_news["content"].fillna("")
embeddings = model.encode(news_texts.tolist())

# 🔹 Convert embeddings to numpy array
embeddings = np.array(embeddings, dtype=np.float32)

# 🔹 Initialize FAISS Index
dimension = embeddings.shape[1]  # Number of dimensions in embeddings
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
index.add(embeddings)  # Add embeddings to the FAISS index

# 🔹 Save FAISS index and news metadata
faiss.write_index(index, "news_faiss.index")
df_news.to_csv("news_metadata.csv", index=False)  # Save metadata

print("✅ FAISS Index and Metadata Saved Successfully!")

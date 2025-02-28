import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load financial news data
csv_path = "news_metadata.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file '{csv_path}' not found!")

df_news = pd.read_csv(csv_path)

# Ensure 'content' column exists
if 'content' not in df_news.columns:
    raise KeyError("❌ 'content' column missing in CSV. Make sure the file contains news article text!")

# Create HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert news articles into embeddings and store in FAISS
vector_store = FAISS.from_texts(df_news['content'].tolist(), embeddings)

# Save FAISS index
vector_store.save_local("news_faiss.index")
print("✅ FAISS index created successfully with HuggingFace embeddings!")

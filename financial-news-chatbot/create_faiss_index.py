import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

csv_path = "news_metadata.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file '{csv_path}' not found!")

df_news = pd.read_csv(csv_path)

if 'content' not in df_news.columns:
    raise KeyError("❌ 'content' column missing in CSV. Make sure the file contains news article text!")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_texts(df_news['content'].tolist(), embeddings)

vector_store.save_local("news_faiss.index")
print("✅ FAISS index created successfully with HuggingFace embeddings!")

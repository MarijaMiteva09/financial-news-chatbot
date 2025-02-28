import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df_news = pd.read_csv("financial_news.csv")  

model = SentenceTransformer("all-MiniLM-L6-v2")

news_texts = df_news["title"] + " " + df_news["content"].fillna("")
embeddings = model.encode(news_texts.tolist())

embeddings = np.array(embeddings, dtype=np.float32)

dimension = embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)  
index.add(embeddings)  

faiss.write_index(index, "news_faiss.index")
df_news.to_csv("news_metadata.csv", index=False)  
print("✅ FAISS Index and Metadata Saved Successfully!")

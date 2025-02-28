import requests
import pandas as pd

API_KEY = "3a0f2c88cf494e2f9d5a7df39e27900c"  
NEWS_URL = "https://newsapi.org/v2/everything"

def fetch_financial_news(query="finance", page_size=10):
    """
    Fetches the latest financial news articles from NewsAPI.

    Parameters:
    - query (str): The search term (default is 'finance').
    - page_size (int): Number of articles to retrieve.

    Returns:
    - pd.DataFrame: A DataFrame containing the news articles.
    """
    params = {
        "q": query,  
        "apiKey": API_KEY,  
        "language": "en",  
        "sortBy": "publishedAt",  
        "pageSize": page_size,  
    }

    response = requests.get(NEWS_URL, params=params)
    
    data = response.json()
    
    if response.status_code != 200:
        print(f"Error: {data.get('message', 'Failed to fetch news')}")
        return pd.DataFrame()

  
    articles = data.get("articles", [])

    df = pd.DataFrame([
        {
            "title": article["title"],
            "content": article["content"],
            "url": article["url"],
            "published_at": article["publishedAt"]
        }
        for article in articles if article["content"] 
    ])

    return df

# 🔥 Fetch news articles and print them
df_news = fetch_financial_news()
df_news.to_csv("financial_news.csv", index=False)  # Save news articles
print("✅ Financial news saved successfully!")


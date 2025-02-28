import requests
import pandas as pd

# 🔑 Replace with your NewsAPI key
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
        "q": query,  # Search for articles related to 'finance'
        "apiKey": API_KEY,  
        "language": "en",  # Get English articles
        "sortBy": "publishedAt",  # Get the most recent articles
        "pageSize": page_size,  # Limit the number of articles
    }

    # 📡 Send request to NewsAPI
    response = requests.get(NEWS_URL, params=params)
    
    # 🔍 Convert response to JSON
    data = response.json()
    
    # 🛑 Error handling: Check if API call was successful
    if response.status_code != 200:
        print(f"Error: {data.get('message', 'Failed to fetch news')}")
        return pd.DataFrame()

    # 📰 Extract articles
    articles = data.get("articles", [])

    # 🗂️ Convert articles to DataFrame
    df = pd.DataFrame([
        {
            "title": article["title"],
            "content": article["content"],
            "url": article["url"],
            "published_at": article["publishedAt"]
        }
        for article in articles if article["content"]  # Exclude empty content
    ])

    return df

# 🔥 Fetch news articles and print them
df_news = fetch_financial_news()
df_news.to_csv("financial_news.csv", index=False)  # Save news articles
print("✅ Financial news saved successfully!")


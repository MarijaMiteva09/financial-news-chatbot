import faiss
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

try:
    vector_store = FAISS.load_local(
        "news_faiss.index", 
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
        allow_dangerous_deserialization=True
    )
    print("FAISS Index Loaded Successfully")
except Exception as e:
    print("Error loading FAISS Index:", e)

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model and Tokenizer loaded successfully.")

def generate_response(query):
    """
    Uses the GPT-2 model to generate a response to a query.
    """
    print(f"🛠️ Generating response for query: {query}")  
    
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    print("🛠️ Encoded Input Shape:", inputs["input_ids"].shape)  

    outputs = model.generate(
        inputs["input_ids"],  
        max_length=500, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7
    )
    print("🛠️ Raw Model Output:", outputs)  

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("📝 Decoded Response:", response)  
    
    return response


def ask_question(query):
    """
    Retrieves the most relevant financial news based on a user query and generates a response using the GPT-2 model.
    """
    print(f"❓ Asking question: {query}")  
    
    docs = vector_store.similarity_search(query, k=5)
    
    if not docs:
        print("❌ No relevant documents found for the query.")  
    else:
        print("✅ Retrieved Documents:") 
        for doc in docs:
            print("-", doc.page_content)  
    
    context = "\n".join([doc.page_content for doc in docs])
    
    augmented_query = f"Context: {context}\nQuery: {query}"
    
    response = generate_response(augmented_query)
    print("Generated Response:", response)  
    return response

def main():
    test_query = "Tell me about stock market trends"
    print(f"📝 Testing query: {test_query}")
    response = ask_question(test_query)
    print("📝 Model Response:", response)


if __name__ == "__main__":
    main()

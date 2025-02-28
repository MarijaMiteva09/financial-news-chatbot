import faiss
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🔹 Load FAISS index
try:
    vector_store = FAISS.load_local(
        "news_faiss.index", 
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
        allow_dangerous_deserialization=True
    )
    print("FAISS Index Loaded Successfully")
except Exception as e:
    print("Error loading FAISS Index:", e)

# 🔹 Use GPT-2 model instead of GPT-4All Falcon
model_name = "gpt2"  # Using the GPT-2 model for testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token to eos_token to prevent the padding error
tokenizer.pad_token = tokenizer.eos_token

# Check if you're using a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model and Tokenizer loaded successfully.")

def generate_response(query):
    """
    Uses the GPT-2 model to generate a response to a query.
    """
    print(f"🛠️ Generating response for query: {query}")  # Debugging line
    
    # Tokenize the input query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    print("🛠️ Encoded Input Shape:", inputs["input_ids"].shape)  # Check if inputs are valid

    # Generate output
    outputs = model.generate(
        inputs["input_ids"],  
        max_length=500, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7
    )
    print("🛠️ Raw Model Output:", outputs)  # Print raw tensor output

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("📝 Decoded Response:", response)  # Check final decoded text
    
    return response


def ask_question(query):
    """
    Retrieves the most relevant financial news based on a user query and generates a response using the GPT-2 model.
    """
    print(f"❓ Asking question: {query}")  # Debugging line
    
    # Retrieve the most relevant documents from FAISS
    docs = vector_store.similarity_search(query, k=5)
    
    if not docs:
        print("❌ No relevant documents found for the query.")  # Debugging line
    else:
        print("✅ Retrieved Documents:")  # Debugging line
        for doc in docs:
            print("-", doc.page_content)  # Debugging line
    
    # Concatenate the retrieved documents to the query as context for the model
    context = "\n".join([doc.page_content for doc in docs])
    
    # Include the context in the query to improve the response
    augmented_query = f"Context: {context}\nQuery: {query}"
    
    # Generate a response based on the augmented query
    response = generate_response(augmented_query)
    print("Generated Response:", response)  # Add this line to check the generated response
    return response


def main():
    # Test the generation with a direct query (without FAISS)
    test_query = "Tell me about stock market trends"
    print(f"📝 Testing query: {test_query}")
    response = ask_question(test_query)
    print("📝 Model Response:", response)


if __name__ == "__main__":
    main()

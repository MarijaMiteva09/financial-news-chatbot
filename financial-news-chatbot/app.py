import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import os

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Replace with your actual model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding

    return model, tokenizer

# Load FAISS Index
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("news_faiss.index/index.faiss")  # Replace with actual file path
    print(f"FAISS index dimension: {index.d}")
    check_faiss_index(index)  # Check the FAISS index properties
    return index

def check_faiss_index(index):
    """Check the properties of the FAISS index."""
    print("FAISS index dimension:", index.d)
    print("Number of vectors in the index:", index.ntotal)

# Encode the query into the same embedding space used for the FAISS index
def encode_query(query, tokenizer, model, faiss_dim=384, max_length=512):
    """
    Encodes the query into a vector for FAISS search.
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get last layer's hidden state
        embeddings = hidden_states.mean(dim=1)  # Average pooling

    query_embedding = embeddings.numpy().astype('float32')

    # Print the query embedding shape for debugging
    print("Query embedding shape:", query_embedding.shape)

    # Ensure the query embedding matches FAISS index dimension
    if query_embedding.shape[1] != faiss_dim:
        print(f"Warning: Query embedding dim {query_embedding.shape[1]} != FAISS dim {faiss_dim}")
        query_embedding = query_embedding[:, :faiss_dim]  # Truncate or pad if needed

    return query_embedding

# Retrieve relevant documents from FAISS index
def retrieve_documents(query, index, tokenizer, model, k=5):
    """
    Encodes the query and retrieves the top K most relevant documents from FAISS.
    """
    query_embedding = encode_query(query, tokenizer, model)

    # Ensure embedding has correct dimensions before FAISS search
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {index.d}")

    D, I = index.search(query_embedding, k)  # Perform FAISS search
    
    # Print the distances and indices for debugging
    print("Distances:", D)
    print("Indices:", I)

    return I  # Return retrieved document indices

# Generate a response based on documents
# Generate a response based on documents
def generate_response(documents, model, tokenizer):
    # Convert document embeddings (NumPy arrays) to strings properly
    documents = [str(doc) for doc in documents]  # Convert each document to a string

    context = " ".join(documents)  # Concatenate documents as context
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate the response with added diversity controls
    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,          # Limit the length of the generated text
        num_beams=5,             # Use beam search to improve response quality
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        temperature=0.7,         # Add randomness to avoid too deterministic output
        top_p=0.9,               # Use top-p sampling to add more variety
        top_k=50                 # Limits the number of top-k tokens to sample from
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load documents (replace with your actual data source)
def load_documents():
    # Replace with actual logic to load your documents, for now, let's use a sample.
    documents = {
        'doc1': {'page_content': 'Stock prices surged today due to strong earnings reports.'},
        'doc2': {'page_content': 'The Federal Reserve announced a new interest rate policy.'},
        'doc3': {'page_content': 'Tech stocks fell sharply after regulatory concerns.'},
        'doc4': {'page_content': 'Investors are optimistic about the future of AI companies.'},
        'doc5': {'page_content': 'Oil prices dropped as supply concerns eased.'}
    }
    
    # Extract document content
    extracted_documents = [doc['page_content'] for doc in documents.values() if 'page_content' in doc]
    return extracted_documents

# Streamlit UI
# Streamlit UI
def main():
    st.title("Stock Market Trends and News Retrieval")

    query = st.text_input("Enter your query about stock market trends:", "Tell me about stock market trends")

    if query:
        # Load model and FAISS index
        model, tokenizer = load_model()
        index = load_faiss_index()

        # Load documents (Replace this with your actual data source)
        documents = load_documents()

        # Retrieve relevant documents from FAISS index
        retrieved_docs = retrieve_documents(query, index, tokenizer, model, k=5)
        st.write(f"Retrieved Document Indices: {retrieved_docs}")

        if retrieved_docs.any():  # Check if any indices are retrieved
            # Get the actual documents based on indices
            retrieved_document_texts = [documents[i] for i in retrieved_docs.flatten() if i < len(documents)]
            st.write(f"Retrieved Documents: {retrieved_document_texts}")

            # Generate response based on documents
            response = generate_response(retrieved_document_texts, model, tokenizer)
            st.write(f"Generated Response: {response}")
        else:
            st.write("No relevant documents retrieved.")


# Run the app
if __name__ == "__main__":
    main()

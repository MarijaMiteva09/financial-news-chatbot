import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import os

@st.cache_resource
def load_model():
    model_name = "gpt2" 
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token  

    return model, tokenizer

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("news_faiss.index/index.faiss") 
    print(f"FAISS index dimension: {index.d}")
    check_faiss_index(index)  
    return index

def check_faiss_index(index):
    """Check the properties of the FAISS index."""
    print("FAISS index dimension:", index.d)
    print("Number of vectors in the index:", index.ntotal)

def encode_query(query, tokenizer, model, faiss_dim=384, max_length=512):
    """
    Encodes the query into a vector for FAISS search.
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  
        embeddings = hidden_states.mean(dim=1)  

    query_embedding = embeddings.numpy().astype('float32')

    print("Query embedding shape:", query_embedding.shape)

    if query_embedding.shape[1] != faiss_dim:
        print(f"Warning: Query embedding dim {query_embedding.shape[1]} != FAISS dim {faiss_dim}")
        query_embedding = query_embedding[:, :faiss_dim] 

    return query_embedding

def retrieve_documents(query, index, tokenizer, model, k=5):
    """
    Encodes the query and retrieves the top K most relevant documents from FAISS.
    """
    query_embedding = encode_query(query, tokenizer, model)

    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {index.d}")

    D, I = index.search(query_embedding, k)  
    
    print("Distances:", D)
    print("Indices:", I)

    return I  

def generate_response(documents, model, tokenizer):
    documents = [str(doc) for doc in documents]  

    context = " ".join(documents)  
    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=512)

    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,         
        num_beams=5,             
        no_repeat_ngram_size=2,  
        temperature=0.7,         
        top_p=0.9,              
        top_k=50                 
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_documents():
    documents = {
        'doc1': {'page_content': 'Stock prices surged today due to strong earnings reports.'},
        'doc2': {'page_content': 'The Federal Reserve announced a new interest rate policy.'},
        'doc3': {'page_content': 'Tech stocks fell sharply after regulatory concerns.'},
        'doc4': {'page_content': 'Investors are optimistic about the future of AI companies.'},
        'doc5': {'page_content': 'Oil prices dropped as supply concerns eased.'}
    }
    
    extracted_documents = [doc['page_content'] for doc in documents.values() if 'page_content' in doc]
    return extracted_documents

def main():
    st.title("Stock Market Trends and News Retrieval")

    query = st.text_input("Enter your query about stock market trends:", "Tell me about stock market trends")

    if query:
        model, tokenizer = load_model()
        index = load_faiss_index()

        documents = load_documents()

        retrieved_docs = retrieve_documents(query, index, tokenizer, model, k=5)
        st.write(f"Retrieved Document Indices: {retrieved_docs}")

        if retrieved_docs.any():  
            retrieved_document_texts = [documents[i] for i in retrieved_docs.flatten() if i < len(documents)]
            st.write(f"Retrieved Documents: {retrieved_document_texts}")

            response = generate_response(retrieved_document_texts, model, tokenizer)
            st.write(f"Generated Response: {response}")
        else:
            st.write("No relevant documents retrieved.")

if __name__ == "__main__":
    main()

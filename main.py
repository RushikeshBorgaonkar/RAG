import os
from db_config import DatabaseManager
from embedding_generator import EmbeddingGenerator
from groq import Groq
from pydantic import BaseModel
from typing import List, Optional
from psycopg2.extras import execute_values
import numpy as np

# Add Pydantic models
class Metadata(BaseModel):
    similarity_score: float

class RetrievedDocument(BaseModel):
    document_id: str
    content: str
    metadata: Metadata

class RetrievalResult(BaseModel):
    query: str
    retrieved_documents: List[RetrievedDocument]
    retrieval_method: str = "vector search"
    generated_response: Optional[str] = None

DATA_FILE_PATH = 'text_samples.txt'

def load_text_samples(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found.")
    with open(file_path, 'r') as file:
        # Remove duplicate lines and empty lines
        texts = list(dict.fromkeys(filter(None, file.read().splitlines())))
    return texts

def generate_augmented_response(query: str, retrieved_items: List[tuple[str, float]]) -> RetrievalResult:
    # Create the retrieval result structure
    retrieved_docs = [
        RetrievedDocument(
            document_id=f"doc_{idx}",
            content=text,
            metadata=Metadata(similarity_score=score)
        )
        for idx, (text, score) in enumerate(retrieved_items)
    ]
    
    context = " ".join(text for text, _ in retrieved_items)
    
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    prompt = f"""Answer the following question using ONLY the information provided in the context below. 
    If the context doesn't contain enough information to fully answer the question, 
    respond with what can be answered from the context only.Context:{context} Question: {query}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": """You are an assistant that provides answers strictly based on the given context."""
                },
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.5,  # Reduced for more consistent, factual responses
            max_tokens=1024
        )

        response = chat_completion.choices[0].message.content.strip()
        
        return RetrievalResult(
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response=response
        )

    except Exception as e:
        print("Error generating response:", str(e))
        return RetrievalResult(
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response="Error generating response."
        )

def main():
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()

    # Clear existing data before adding new embeddings
    db_manager.clear_embeddings()

    texts = load_text_samples(DATA_FILE_PATH)
    # Add check for duplicate texts
    processed_texts = set()
    for idx, text in enumerate(texts):
        if text in processed_texts:
            continue
        processed_texts.add(text)
        
        embedding = embedding_gen.generate_embedding(text)
        db_manager.add_embedding_to_db(embedding, text_id=str(idx), text_content=text)
        print("------------------------------------------------------------------------------------------------")
        print(f"Added embedding for: {text}")
        print(f"Embeddings for above text : {embedding}")
        print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")
    
    query_text = "History about ram temple"
    query_embedding = embedding_gen.generate_embedding(query_text)
    print(f"query_text: {query_text}")
    print(f"Embedding query_text :{query_embedding}")
    print("------------------------------------------------------------------------------------------------")
    
    similar_items = db_manager.search_similar_vectors(query_embedding, top_k=3)
    
    print(f"similar_items found are: {similar_items}")
    print("------------------------------------------------------------------------------------------------")
    
    # Get the structured response
    result = generate_augmented_response(query_text, similar_items)
    
    # Print the response in JSON format
    print("Structured Response:")
    print(result.model_dump_json(indent=2))

    db_manager.close()

if __name__ == "__main__":
    main()

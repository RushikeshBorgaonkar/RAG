import os
from db_config import DatabaseManager
from embedding_generator import EmbeddingGenerator
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
from psycopg2.extras import execute_values
import numpy as np

# Add Pydantic models
class Metadata(BaseModel):
    similarity_score: float = Field(
        gt=0.0,  # greater than 0
        le=1.0,  # less than or equal to 1
        description="Cosine similarity score between query and document",
        example=0.85
    )

class RetrievedDocument(BaseModel):
    document_id: str = Field(
        min_length=1,
        description="Unique identifier for the document",
        example="doc_1"
    )
    content: str = Field(
        min_length=1,
        description="The actual text content of the document",
        example="This is a sample document content"
    )
    metadata: Metadata = Field(
        description="Additional metadata about the document including similarity score"
    )

class RetrievalResult(BaseModel):
    query: str = Field(
        min_length=1,
        description="The original search query",
        example="History about ram temple"
    )
    retrieved_documents: List[RetrievedDocument] = Field(
        description="List of retrieved documents with their metadata",
        min_items=0,  # Allow empty list
        max_items=10  # Maximum 10 documents
    )
    retrieval_method: str = Field(
        default="vector search",
        description="Method used for retrieving documents",
        example="vector search"
    )
    generated_response: Optional[str] = Field(
        default=None,
        description="AI-generated response based on retrieved documents",
        example="Based on the retrieved documents..."
    )


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
    
    # Format context with numbered documents for clarity
    context = "\n\n".join(f"Document {idx + 1}:\n{text}" 
                         for idx, (text, _) in enumerate(retrieved_items))
    
    prompt = f"""Please provide a comprehensive response to the question using information from ALL the provided documents below. 
    Synthesize information from all available documents to create a complete answer.
    
    Context Documents:
    {context}

    Question: {query}

    Instructions:
    1. Use information from all provided documents
    2. Synthesize the information into a coherent response
    3. Include relevant details from each document
    4. Maintain accuracy and context relevance
    5. If there are any contradictions between documents, mention them
    
    Please provide a detailed response incorporating information from all documents.
    """

    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": """You are an assistant that provides comprehensive answers by analyzing and 
                    synthesizing information from multiple documents. Always strive to include relevant 
                    information from all provided documents in your response."""
                },
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1024   # Increased max tokens for longer responses
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
    
    query_text = "Tell me architectural features of Ram Mandir"
    query_embedding = embedding_gen.generate_embedding(query_text)
    print(f"query_text: {query_text}")
    print(f"Embedding query_text :{query_embedding}")
    print("------------------------------------------------------------------------------------------------")
    
    similar_items = db_manager.search_similar_vectors(query_embedding, top_k=2)
    
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

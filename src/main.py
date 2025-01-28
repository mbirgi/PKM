import os
import weaviate
from weaviate.connect import ConnectionParams
import torch
from sentence_transformers import SentenceTransformer

class DocumentSearchEngine:
    def __init__(self, weaviate_url='http://localhost:8080', embedding_model='all-MiniLM-L6-v2'):
        # Initialize Weaviate client with ConnectionParams
        # connection_params = ConnectionParams(url=weaviate_url)
        self.client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create Weaviate schema
        self._create_schema()
    
    def _create_schema(self):
        # Define document schema for Weaviate
        schema = {
            "classes": [
                {
                    "class": "Document",
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "metadata", "dataType": ["text"]},
                        {"name": "embedding", "dataType": ["number[]"]}
                    ]
                }
            ]
        }
        
        # Create schema if not exists
        if not self.client.schema.exists("Document"):
            self.client.schema.create(schema)
    
    def index_document(self, content, source, metadata=None):
        # Generate embedding
        embedding = self.embedding_model.encode(content)
        
        # Prepare document object
        document = {
            "content": content,
            "source": source,
            "metadata": metadata,
            "embedding": embedding.tolist()  # Convert embedding to list
        }
        
        # Index document in Weaviate
        self.client.data_object.create(document, "Document")
    
    def search(self, query, top_k=5):
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Perform vector search
        result = self.client.query.get(
            "Document", 
            ["content", "source", "metadata"]
        ).with_near_vector({
            "vector": query_embedding.tolist(),
            "distance": 0.7
        }).with_limit(top_k).do()
        
        return result['data']['Get']['Document']

# Example usage
if __name__ == "__main__":
    search_engine = DocumentSearchEngine()
    search_engine.index_document("This is a test document.", "test_source")
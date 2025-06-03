"""
Vector Store for Email Embeddings
Stores and retrieves email thread embeddings using FAISS or Weaviate
"""

import os
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_store")

# Try to import FAISS, with fallback to using pickle for storage
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available for vector storage")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to basic storage")
    import pickle

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("SentenceTransformer is available for embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformer not available, using random embeddings for demo")

# Try to import Weaviate client
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
    logger.info("Weaviate client is available")
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate client not available")


class VectorStore:
    """
    Vector store for email embeddings using FAISS or Weaviate
    Allows for semantic search and retrieval of similar emails
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 storage_type: str = "faiss",
                 dimension: int = 384,
                 data_dir: str = "./data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            storage_type: Type of vector storage ('faiss' or 'weaviate')
            dimension: Dimension of the embedding vectors
            data_dir: Directory to store the vector data
        """
        self.embedding_model_name = embedding_model
        self.storage_type = storage_type
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Initialized embedding model: {embedding_model}")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("No embedding model available")
        
        # Initialize vector storage
        self._initialize_storage()
        
        # Load existing data if available
        self.email_data = self._load_email_data()
        self.next_id = len(self.email_data)
        
        logger.info(f"VectorStore initialized with {self.next_id} existing emails")
    
    def _initialize_storage(self):
        """Initialize the vector storage based on the selected type"""
        if self.storage_type == "faiss" and FAISS_AVAILABLE:
            # Initialize FAISS index
            self.index_path = self.data_dir / "faiss_index.bin"
            if self.index_path.exists():
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"Loaded existing FAISS index from {self.index_path}")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")
                    self._create_new_faiss_index()
            else:
                self._create_new_faiss_index()
        
        elif self.storage_type == "weaviate" and WEAVIATE_AVAILABLE:
            # Initialize Weaviate client
            try:
                self.weaviate_client = weaviate.Client(
                    url=os.getenv("WEAVIATE_URL", "http://localhost:8080")
                )
                
                # Check if schema exists, if not create it
                self._ensure_weaviate_schema()
                logger.info("Connected to Weaviate instance")
            except Exception as e:
                logger.error(f"Error connecting to Weaviate: {e}")
                # Fallback to FAISS
                self.storage_type = "faiss"
                self._create_new_faiss_index()
        
        else:
            # Fallback to basic storage if neither is available
            logger.warning(f"Storage type {self.storage_type} not available, falling back to basic storage")
            self.storage_type = "basic"
            self.vectors_path = self.data_dir / "vectors.pkl"
            if self.vectors_path.exists():
                try:
                    with open(self.vectors_path, 'rb') as f:
                        self.vectors = pickle.load(f)
                    logger.info(f"Loaded existing vectors from {self.vectors_path}")
                except Exception as e:
                    logger.error(f"Error loading vectors: {e}")
                    self.vectors = []
            else:
                self.vectors = []
    
    def _create_new_faiss_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _ensure_weaviate_schema(self):
        """Ensure the Weaviate schema exists"""
        if WEAVIATE_AVAILABLE:
            # Define the email schema
            schema = {
                "classes": [{
                    "class": "Email",
                    "description": "An email with embeddings for semantic search",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "subject",
                            "dataType": ["string"],
                            "description": "The email subject"
                        },
                        {
                            "name": "body",
                            "dataType": ["text"],
                            "description": "The email body content"
                        },
                        {
                            "name": "sender",
                            "dataType": ["string"],
                            "description": "The email sender"
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["date"],
                            "description": "When the email was sent or received"
                        },
                        {
                            "name": "cluster",
                            "dataType": ["string"],
                            "description": "The cluster this email belongs to"
                        },
                        {
                            "name": "sentiment",
                            "dataType": ["string"],
                            "description": "The sentiment of the email"
                        }
                    ]
                }]
            }
            
            # Check if schema exists
            try:
                current_schema = self.weaviate_client.schema.get()
                existing_classes = [c["class"] for c in current_schema["classes"]]
                
                if "Email" not in existing_classes:
                    # Create schema
                    self.weaviate_client.schema.create(schema)
                    logger.info("Created Email schema in Weaviate")
            except Exception as e:
                logger.error(f"Error creating Weaviate schema: {e}")
    
    def _load_email_data(self) -> Dict[int, Dict]:
        """Load email metadata from disk"""
        metadata_path = self.data_dir / "email_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading email metadata: {e}")
                return {}
        return {}
    
    def _save_email_data(self):
        """Save email metadata to disk"""
        metadata_path = self.data_dir / "email_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.email_data, f)
        except Exception as e:
            logger.error(f"Error saving email metadata: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode(text)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
        
        # Fallback to random embedding for demo purposes
        logger.warning("Using random embedding as fallback")
        return np.random.rand(self.dimension).astype(np.float32)
    
    def store_email(self, email_data: Dict[str, Any]) -> int:
        """
        Store an email in the vector database
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments', 'cluster', 'sentiment'
        
        Returns:
            email_id: ID of the stored email
        """
        # Log the operation
        logger.info(f"Storing email: {email_data.get('subject', 'No subject')}")
        
        # Generate embedding from subject and body
        text_to_embed = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
        embedding = self._get_embedding(text_to_embed)
        
        # Store in the appropriate backend
        email_id = self.next_id
        
        if self.storage_type == "faiss" and FAISS_AVAILABLE:
            # Add to FAISS index
            self.index.add(np.array([embedding]).astype('float32'))
            # Save index periodically
            if email_id % 10 == 0:
                faiss.write_index(self.index, str(self.index_path))
        
        elif self.storage_type == "weaviate" and WEAVIATE_AVAILABLE:
            # Add to Weaviate
            try:
                # Convert datetime to string if needed
                email_props = email_data.copy()
                if isinstance(email_props.get('timestamp'), datetime):
                    email_props['timestamp'] = email_props['timestamp'].isoformat()
                
                # Remove attachments as they might not be serializable
                if 'attachments' in email_props:
                    del email_props['attachments']
                
                # Add to Weaviate with the embedding
                self.weaviate_client.data_object.create(
                    "Email",
                    email_props,
                    vector=embedding.tolist()
                )
            except Exception as e:
                logger.error(f"Error storing in Weaviate: {e}")
        
        else:
            # Basic storage
            self.vectors.append(embedding)
            # Save vectors periodically
            if email_id % 10 == 0 and hasattr(self, 'vectors_path'):
                with open(self.vectors_path, 'wb') as f:
                    pickle.dump(self.vectors, f)
        
        # Store metadata
        self.email_data[str(email_id)] = {
            'subject': email_data.get('subject', 'No subject'),
            'sender': email_data.get('sender', 'Unknown'),
            'timestamp': (email_data.get('timestamp', datetime.now()).isoformat() 
                         if isinstance(email_data.get('timestamp'), datetime) 
                         else str(email_data.get('timestamp', 'Unknown'))),
            'cluster': email_data.get('cluster', 'Uncategorized'),
            'sentiment': email_data.get('sentiment', 'Neutral')
        }
        
        # Save metadata
        self._save_email_data()
        
        # Increment ID counter
        self.next_id += 1
        
        return email_id
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for emails similar to the query
        
        Args:
            query: Text query to search for
            k: Number of results to return
        
        Returns:
            List of dictionaries with email data and similarity scores
        """
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        results = []
        
        if self.storage_type == "faiss" and FAISS_AVAILABLE:
            # Search in FAISS
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty, no results")
                return []
            
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                min(k, self.index.ntotal)
            )
            
            # Format results
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if str(idx) in self.email_data:
                    email_info = self.email_data[str(idx)]
                    results.append({
                        'id': idx,
                        'similarity': 1.0 / (1.0 + distance),  # Convert distance to similarity
                        **email_info
                    })
        
        elif self.storage_type == "weaviate" and WEAVIATE_AVAILABLE:
            # Search in Weaviate
            try:
                weaviate_results = (
                    self.weaviate_client.query
                    .get("Email", ["subject", "sender", "timestamp", "cluster", "sentiment"])
                    .with_near_vector({"vector": query_embedding.tolist()})
                    .with_limit(k)
                    .do()
                )
                
                # Format results
                if weaviate_results and "data" in weaviate_results:
                    for item in weaviate_results["data"]["Get"]["Email"]:
                        results.append({
                            'id': item.get("_additional", {}).get("id", "unknown"),
                            'similarity': item.get("_additional", {}).get("certainty", 0.0),
                            **item
                        })
            except Exception as e:
                logger.error(f"Error searching in Weaviate: {e}")
        
        else:
            # Basic search with numpy
            if not self.vectors:
                logger.warning("No vectors stored, no results")
                return []
            
            # Calculate distances
            vectors = np.array(self.vectors)
            distances = np.linalg.norm(vectors - query_embedding, axis=1)
            
            # Get top k indices
            top_indices = np.argsort(distances)[:min(k, len(distances))]
            
            # Format results
            for i, idx in enumerate(top_indices):
                if str(idx) in self.email_data:
                    email_info = self.email_data[str(idx)]
                    results.append({
                        'id': idx,
                        'similarity': 1.0 / (1.0 + distances[idx]),  # Convert distance to similarity
                        **email_info
                    })
        
        return results
    
    def get_email_clusters(self) -> Dict[str, int]:
        """
        Get counts of emails by cluster
        
        Returns:
            Dictionary mapping cluster names to counts
        """
        clusters = {}
        for email_id, email_info in self.email_data.items():
            cluster = email_info.get('cluster', 'Uncategorized')
            clusters[cluster] = clusters.get(cluster, 0) + 1
        
        return clusters
    
    def get_email_by_id(self, email_id: int) -> Optional[Dict[str, Any]]:
        """
        Get email data by ID
        
        Args:
            email_id: ID of the email to retrieve
        
        Returns:
            Email data dictionary or None if not found
        """
        return self.email_data.get(str(email_id))
    
    def delete_email(self, email_id: int) -> bool:
        """
        Delete an email from the vector store
        
        Args:
            email_id: ID of the email to delete
        
        Returns:
            Success flag
        """
        # Note: Actual deletion from FAISS requires rebuilding the index
        # For simplicity, we just remove from metadata
        if str(email_id) in self.email_data:
            del self.email_data[str(email_id)]
            self._save_email_data()
            logger.info(f"Deleted email ID {email_id} from metadata")
            return True
        return False
    
    def clear_all(self) -> bool:
        """
        Clear all data from the vector store
        
        Returns:
            Success flag
        """
        try:
            # Clear metadata
            self.email_data = {}
            self._save_email_data()
            
            # Reset storage based on type
            if self.storage_type == "faiss" and FAISS_AVAILABLE:
                self._create_new_faiss_index()
                faiss.write_index(self.index, str(self.index_path))
            
            elif self.storage_type == "weaviate" and WEAVIATE_AVAILABLE:
                # Delete all objects of class Email
                self.weaviate_client.schema.delete_class("Email")
                # Recreate schema
                self._ensure_weaviate_schema()
            
            else:
                # Basic storage
                self.vectors = []
                if hasattr(self, 'vectors_path'):
                    with open(self.vectors_path, 'wb') as f:
                        pickle.dump(self.vectors, f)
            
            self.next_id = 0
            logger.info("Cleared all data from vector store")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

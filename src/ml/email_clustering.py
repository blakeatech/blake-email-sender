"""
Advanced Email Clustering and Categorization System
Uses sentence transformers, UMAP, and HDBSCAN for intelligent email grouping
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import logging
import pickle
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailClusteringSystem:
    """Advanced email clustering using neural embeddings and unsupervised learning"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the clustering system
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        self.umap_model = None
        self.hdbscan_model = None
        self.tfidf_vectorizer = None
        self.cluster_labels = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.email_data = []
        
        logger.info(f"Initialized EmailClusteringSystem with model: {model_name}")
    
    def add_emails(self, emails: List[Dict]):
        """
        Add emails to the clustering system
        
        Args:
            emails: List of email dictionaries with 'subject', 'body', 'sender', 'timestamp'
        """
        for email in emails:
            # Combine subject and body for clustering
            full_text = f"{email.get('subject', '')} {email.get('body', '')}"
            
            email_entry = {
                'id': len(self.email_data),
                'subject': email.get('subject', ''),
                'body': email.get('body', ''),
                'sender': email.get('sender', ''),
                'timestamp': email.get('timestamp', datetime.now()),
                'full_text': full_text.strip(),
                'cluster': None
            }
            
            self.email_data.append(email_entry)
        
        logger.info(f"Added {len(emails)} emails. Total emails: {len(self.email_data)}")
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate sentence embeddings for all emails
        
        Returns:
            Array of embeddings
        """
        if not self.email_data:
            raise ValueError("No emails added to the system")
        
        texts = [email['full_text'] for email in self.email_data]
        
        logger.info("Generating sentence embeddings...")
        self.embeddings = self.sentence_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def reduce_dimensions(self, n_components: int = 2, n_neighbors: int = 15, 
                         min_dist: float = 0.1) -> np.ndarray:
        """
        Reduce dimensionality using UMAP
        
        Args:
            n_components: Number of dimensions to reduce to
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            
        Returns:
            Reduced embeddings
        """
        if self.embeddings is None:
            self.generate_embeddings()
        
        logger.info("Reducing dimensions with UMAP...")
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        
        self.reduced_embeddings = self.umap_model.fit_transform(self.embeddings)
        logger.info(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")
        
        return self.reduced_embeddings
    
    def cluster_emails(self, min_cluster_size: int = 5, min_samples: int = 3) -> np.ndarray:
        """
        Cluster emails using HDBSCAN
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples for core points
            
        Returns:
            Cluster labels
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        logger.info("Clustering emails with HDBSCAN...")
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        self.cluster_labels = self.hdbscan_model.fit_predict(self.reduced_embeddings)
        
        # Update email data with cluster labels
        for i, email in enumerate(self.email_data):
            email['cluster'] = int(self.cluster_labels[i])
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return self.cluster_labels
    
    def analyze_clusters(self) -> Dict:
        """
        Analyze the characteristics of each cluster
        
        Returns:
            Dictionary with cluster analysis
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet")
        
        cluster_analysis = {}
        unique_clusters = set(self.cluster_labels)
        
        # Generate TF-IDF for keyword extraction
        texts = [email['full_text'] for email in self.email_data]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_emails = [email for email in self.email_data if email['cluster'] == cluster_id]
            cluster_indices = [i for i, email in enumerate(self.email_data) if email['cluster'] == cluster_id]
            
            # Extract top keywords for this cluster
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            top_keyword_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_keyword_indices]
            
            # Analyze senders
            senders = [email['sender'] for email in cluster_emails]
            sender_counts = pd.Series(senders).value_counts()
            
            # Analyze subjects
            subjects = [email['subject'] for email in cluster_emails]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_emails),
                'top_keywords': top_keywords,
                'top_senders': sender_counts.head(5).to_dict(),
                'sample_subjects': subjects[:5],
                'avg_text_length': np.mean([len(email['full_text']) for email in cluster_emails]),
                'emails': cluster_emails
            }
        
        return cluster_analysis
    
    def suggest_categories(self) -> Dict[int, str]:
        """
        Suggest category names for each cluster based on keywords
        
        Returns:
            Dictionary mapping cluster IDs to suggested category names
        """
        cluster_analysis = self.analyze_clusters()
        category_suggestions = {}
        
        # Predefined category mappings based on keywords
        category_keywords = {
            'Technical Support': ['error', 'bug', 'issue', 'problem', 'help', 'support', 'fix'],
            'Meeting Requests': ['meeting', 'schedule', 'calendar', 'appointment', 'call', 'zoom'],
            'Project Updates': ['project', 'update', 'progress', 'status', 'milestone', 'deadline'],
            'Sales Inquiries': ['price', 'quote', 'purchase', 'buy', 'cost', 'sales', 'product'],
            'Collaboration': ['collaborate', 'partnership', 'work together', 'team', 'joint'],
            'Information Requests': ['information', 'details', 'question', 'inquiry', 'clarification'],
            'Complaints': ['complaint', 'dissatisfied', 'unhappy', 'problem', 'issue', 'disappointed'],
            'Introductions': ['introduction', 'hello', 'nice to meet', 'new', 'introduce'],
            'Follow-ups': ['follow up', 'following up', 'checking in', 'update', 'status'],
            'Urgent Matters': ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        }
        
        for cluster_id, analysis in cluster_analysis.items():
            top_keywords = [kw.lower() for kw in analysis['top_keywords']]
            
            best_category = 'Miscellaneous'
            best_score = 0
            
            for category, keywords in category_keywords.items():
                score = sum(1 for kw in keywords if any(kw in top_kw for top_kw in top_keywords))
                if score > best_score:
                    best_score = score
                    best_category = category
            
            category_suggestions[cluster_id] = best_category
        
        return category_suggestions
    
    def visualize_clusters(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive visualization of the clusters
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if self.reduced_embeddings is None or self.cluster_labels is None:
            raise ValueError("Clustering must be performed before visualization")
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': self.cluster_labels,
            'subject': [email['subject'] for email in self.email_data],
            'sender': [email['sender'] for email in self.email_data]
        })
        
        # Create the plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='cluster',
            hover_data=['subject', 'sender'],
            title='Email Clusters Visualization',
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def predict_cluster(self, new_email: Dict) -> Tuple[int, float]:
        """
        Predict the cluster for a new email
        
        Args:
            new_email: Dictionary with 'subject' and 'body'
            
        Returns:
            Tuple of (predicted_cluster, confidence)
        """
        if self.embeddings is None:
            raise ValueError("Model must be trained before prediction")
        
        # Generate embedding for new email
        full_text = f"{new_email.get('subject', '')} {new_email.get('body', '')}"
        new_embedding = self.sentence_model.encode([full_text])
        
        # Transform with UMAP
        new_reduced = self.umap_model.transform(new_embedding)
        
        # Find closest cluster
        cluster_centers = {}
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            cluster_points = self.reduced_embeddings[self.cluster_labels == cluster_id]
            cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
        
        # Calculate distances to cluster centers
        distances = {}
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(new_reduced[0] - center)
            distances[cluster_id] = distance
        
        # Find closest cluster
        closest_cluster = min(distances, key=distances.get)
        confidence = 1.0 / (1.0 + distances[closest_cluster])  # Convert distance to confidence
        
        return closest_cluster, confidence
    
    def save_model(self, filepath: str):
        """Save the trained clustering model"""
        model_data = {
            'umap_model': self.umap_model,
            'hdbscan_model': self.hdbscan_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'embeddings': self.embeddings,
            'reduced_embeddings': self.reduced_embeddings,
            'cluster_labels': self.cluster_labels,
            'email_data': self.email_data,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained clustering model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.umap_model = model_data['umap_model']
        self.hdbscan_model = model_data['hdbscan_model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.embeddings = model_data['embeddings']
        self.reduced_embeddings = model_data['reduced_embeddings']
        self.cluster_labels = model_data['cluster_labels']
        self.email_data = model_data['email_data']
        self.model_name = model_data['model_name']
        
        # Reinitialize sentence transformer
        self.sentence_model = SentenceTransformer(self.model_name)
        
        logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample email data
    sample_emails = [
        {
            'subject': 'Bug Report: Login Issue',
            'body': 'I am experiencing a login issue with the application. The error message says invalid credentials.',
            'sender': 'user1@example.com',
            'timestamp': datetime.now()
        },
        {
            'subject': 'Meeting Request: Project Discussion',
            'body': 'Could we schedule a meeting to discuss the upcoming project milestones?',
            'sender': 'manager@example.com',
            'timestamp': datetime.now()
        },
        {
            'subject': 'Sales Inquiry: Product Pricing',
            'body': 'I am interested in your product and would like to know about pricing options.',
            'sender': 'customer@example.com',
            'timestamp': datetime.now()
        },
        # Add more sample emails...
    ]
    
    # Initialize and run clustering
    clustering_system = EmailClusteringSystem()
    clustering_system.add_emails(sample_emails)
    clustering_system.generate_embeddings()
    clustering_system.reduce_dimensions()
    clustering_system.cluster_emails()
    
    # Analyze results
    analysis = clustering_system.analyze_clusters()
    categories = clustering_system.suggest_categories()
    
    print("Cluster Analysis:")
    print("=" * 50)
    for cluster_id, data in analysis.items():
        print(f"Cluster {cluster_id} ({categories.get(cluster_id, 'Unknown')}):")
        print(f"  Size: {data['size']}")
        print(f"  Keywords: {data['top_keywords'][:5]}")
        print(f"  Sample subjects: {data['sample_subjects'][:3]}")
        print()
    
    # Visualize
    fig = clustering_system.visualize_clusters()
    fig.show() 
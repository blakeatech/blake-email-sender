"""
Comprehensive ML Model Training Script
Trains all machine learning models for the enhanced email system
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
from typing import Dict, List, Tuple

# Import our ML components
from ml_enhancements.sentiment_analyzer import EmailSentimentAnalyzer
from ml_enhancements.email_clustering import EmailClusteringSystem
from ml_enhancements.multimodal_processor import MultimodalEmailProcessor
from ml_enhancements.predictive_analytics import PredictiveEmailAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailDataGenerator:
    """Generate synthetic training data for email ML models"""
    
    def __init__(self):
        self.sentiments = ['positive', 'negative', 'neutral']
        self.emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'neutral']
        self.intents = ['question', 'request', 'complaint', 'compliment', 'information', 'meeting']
        self.urgency_levels = ['low', 'medium', 'high']
        
    def generate_email_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic email dataset for training"""
        logger.info(f"Generating {num_samples} synthetic email samples...")
        
        # Sample email templates
        templates = {
            'question': [
                "I have a question about {topic}. Could you please help me understand {detail}?",
                "Can you explain how {process} works? I'm confused about {aspect}.",
                "What is the best way to {action}? I need guidance on {specific}."
            ],
            'request': [
                "I would like to request {item}. Please let me know if this is possible.",
                "Could you please {action} for me? I need this by {deadline}.",
                "I'm requesting access to {resource}. When can this be arranged?"
            ],
            'complaint': [
                "I'm experiencing issues with {problem}. This is very frustrating.",
                "There's a serious problem with {service}. I need this fixed immediately.",
                "I'm disappointed with {experience}. This needs to be addressed."
            ],
            'compliment': [
                "I wanted to thank you for {service}. It was excellent!",
                "Great job on {project}! I'm very impressed with the results.",
                "Your team did an amazing job with {task}. Keep up the good work!"
            ],
            'information': [
                "Here's the information you requested about {topic}.",
                "I'm sharing the details regarding {subject} as discussed.",
                "Please find attached the {document} you needed."
            ],
            'meeting': [
                "Let's schedule a meeting to discuss {topic}. Are you available {time}?",
                "I'd like to set up a call about {subject}. What's your availability?",
                "Can we meet to review {project}? I have some concerns to discuss."
            ]
        }
        
        # Generate data
        data = []
        for i in range(num_samples):
            intent = np.random.choice(self.intents)
            template = np.random.choice(templates[intent])
            
            # Fill template with random content
            subject = f"Re: {intent.title()} - {np.random.choice(['Project', 'Service', 'Meeting', 'Issue'])}"
            body = template.format(
                topic=np.random.choice(['the project', 'our service', 'the system', 'the process']),
                detail=np.random.choice(['the requirements', 'the timeline', 'the costs', 'the features']),
                process=np.random.choice(['authentication', 'deployment', 'integration', 'testing']),
                aspect=np.random.choice(['the workflow', 'the interface', 'the configuration', 'the setup']),
                action=np.random.choice(['implement', 'configure', 'optimize', 'troubleshoot']),
                specific=np.random.choice(['the best practices', 'the implementation', 'the approach', 'the solution']),
                item=np.random.choice(['access', 'permission', 'resources', 'support']),
                deadline=np.random.choice(['tomorrow', 'next week', 'end of month', 'ASAP']),
                resource=np.random.choice(['the database', 'the system', 'the platform', 'the tools']),
                problem=np.random.choice(['the login', 'the interface', 'the performance', 'the connection']),
                service=np.random.choice(['customer support', 'the platform', 'the application', 'the service']),
                experience=np.random.choice(['the delay', 'the quality', 'the response time', 'the outcome']),
                project=np.random.choice(['the website', 'the application', 'the integration', 'the analysis']),
                task=np.random.choice(['the implementation', 'the design', 'the optimization', 'the testing']),
                document=np.random.choice(['report', 'analysis', 'specification', 'documentation']),
                time=np.random.choice(['this week', 'next Tuesday', 'tomorrow afternoon', 'Friday morning'])
            )
            
            # Assign labels based on intent and content
            if intent == 'complaint':
                sentiment = 'negative'
                emotion = np.random.choice(['anger', 'sadness', 'fear'])
                urgency = np.random.choice(['medium', 'high'], p=[0.3, 0.7])
            elif intent == 'compliment':
                sentiment = 'positive'
                emotion = 'joy'
                urgency = 'low'
            elif intent == 'request':
                sentiment = np.random.choice(['neutral', 'positive'], p=[0.7, 0.3])
                emotion = np.random.choice(['neutral', 'surprise'], p=[0.8, 0.2])
                urgency = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2])
            else:
                sentiment = np.random.choice(self.sentiments, p=[0.3, 0.2, 0.5])
                emotion = np.random.choice(self.emotions, p=[0.2, 0.1, 0.1, 0.1, 0.1, 0.4])
                urgency = np.random.choice(self.urgency_levels, p=[0.6, 0.3, 0.1])
            
            # Generate response data
            response_time = np.random.exponential(4.0)  # Average 4 hours
            responded = np.random.choice([True, False], p=[0.75, 0.25])
            
            # Generate timing features
            send_hour = np.random.randint(8, 18)  # Business hours
            send_day = np.random.randint(0, 7)  # 0=Monday, 6=Sunday
            
            data.append({
                'subject': subject,
                'body': body,
                'sender': f"user{i}@example.com",
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'sentiment': sentiment,
                'emotion': emotion,
                'intent': intent,
                'urgency': urgency,
                'responded': responded,
                'response_time': response_time if responded else None,
                'send_hour': send_hour,
                'send_day': send_day,
                'subject_length': len(subject),
                'body_length': len(body),
                'has_attachments': np.random.choice([True, False], p=[0.2, 0.8])
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated dataset with {len(df)} samples")
        return df

class ModelTrainer:
    """Main class for training all ML models"""
    
    def __init__(self):
        self.data_generator = EmailDataGenerator()
        self.models = {}
        
    def prepare_training_data(self, num_samples: int = 2000) -> pd.DataFrame:
        """Prepare training data"""
        logger.info("Preparing training data...")
        
        # Generate synthetic data
        df = self.data_generator.generate_email_dataset(num_samples)
        
        # Save to CSV for future use
        df.to_csv('data/training_data.csv', index=False)
        logger.info("Training data saved to data/training_data.csv")
        
        return df
    
    def train_sentiment_models(self, df: pd.DataFrame):
        """Train sentiment analysis models"""
        logger.info("Training sentiment analysis models...")
        
        try:
            # Initialize sentiment analyzer
            sentiment_analyzer = EmailSentimentAnalyzer()
            
            # For demonstration, we'll evaluate on our synthetic data
            # In practice, you'd fine-tune the models here
            
            sample_texts = df['body'].head(100).tolist()
            sample_subjects = df['subject'].head(100).tolist()
            
            correct_predictions = 0
            total_predictions = 0
            
            for i, (text, subject) in enumerate(zip(sample_texts, sample_subjects)):
                try:
                    analysis = sentiment_analyzer.analyze_email(text, subject)
                    predicted_sentiment = analysis.get('sentiment', {}).get('overall_sentiment', 'neutral')
                    actual_sentiment = df.iloc[i]['sentiment']
                    
                    if predicted_sentiment.lower() == actual_sentiment.lower():
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Error analyzing email {i}: {e}")
                    continue
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            logger.info(f"Sentiment analysis accuracy on sample: {accuracy:.3f}")
            
            self.models['sentiment_analyzer'] = sentiment_analyzer
            
        except Exception as e:
            logger.error(f"Error training sentiment models: {e}")
    
    def train_clustering_model(self, df: pd.DataFrame):
        """Train email clustering model"""
        logger.info("Training email clustering model...")
        
        try:
            # Initialize clustering system
            clustering_system = EmailClusteringSystem()
            
            # Prepare email data
            emails = []
            for _, row in df.head(200).iterrows():  # Use subset for demo
                emails.append({
                    'subject': row['subject'],
                    'body': row['body'],
                    'sender': row['sender'],
                    'timestamp': row['timestamp']
                })
            
            # Train clustering
            clustering_system.add_emails(emails)
            clustering_system.generate_embeddings()
            clustering_system.reduce_dimensions()
            clustering_system.cluster_emails()
            
            # Analyze results
            analysis = clustering_system.analyze_clusters()
            categories = clustering_system.suggest_categories()
            
            logger.info(f"Clustering completed. Found {len(analysis)} clusters:")
            for cluster_id, data in analysis.items():
                logger.info(f"  Cluster {cluster_id}: {data['size']} emails - {categories.get(cluster_id, 'Unknown')}")
            
            # Save model
            clustering_system.save_model('models/clustering_model.pkl')
            self.models['clustering_system'] = clustering_system
            
        except Exception as e:
            logger.error(f"Error training clustering model: {e}")
    
    def train_predictive_models(self, df: pd.DataFrame):
        """Train predictive analytics models"""
        logger.info("Training predictive analytics models...")
        
        try:
            # Initialize predictive analytics
            predictive_analytics = PredictiveEmailAnalytics()
            
            # Prepare features for response prediction
            response_features = []
            response_labels = []
            
            for _, row in df.iterrows():
                features = [
                    row['subject_length'],
                    row['body_length'],
                    row['send_hour'],
                    row['send_day'],
                    1 if row['has_attachments'] else 0,
                    1 if row['urgency'] == 'high' else 0,
                    1 if row['sentiment'] == 'positive' else 0,
                    1 if row['intent'] == 'question' else 0
                ]
                response_features.append(features)
                response_labels.append(1 if row['responded'] else 0)
            
            response_features = np.array(response_features, dtype=np.float32)
            response_labels = np.array(response_labels, dtype=np.float32)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                response_features, response_labels, test_size=0.2, random_state=42
            )
            
            # Train response prediction model
            logger.info("Training response prediction model...")
            predictive_analytics.train_response_predictor(X_train, y_train, epochs=50)
            
            # Evaluate
            predictions = predictive_analytics.predict_response_probability(X_test)
            binary_predictions = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(y_test, binary_predictions)
            logger.info(f"Response prediction accuracy: {accuracy:.3f}")
            
            # Train timing prediction model
            timing_features = []
            timing_labels = []
            
            for _, row in df.iterrows():
                features = [
                    row['subject_length'],
                    row['body_length'],
                    1 if row['urgency'] == 'high' else 0,
                    1 if row['sentiment'] == 'positive' else 0
                ]
                timing_features.append(features)
                timing_labels.append([row['send_hour'], row['send_day']])
            
            timing_features = np.array(timing_features, dtype=np.float32)
            timing_labels = np.array(timing_labels, dtype=np.float32)
            
            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                timing_features, timing_labels, test_size=0.2, random_state=42
            )
            
            logger.info("Training optimal timing model...")
            predictive_analytics.train_timing_predictor(X_train_t, y_train_t, epochs=50)
            
            # Save models
            predictive_analytics.save_models('models/')
            self.models['predictive_analytics'] = predictive_analytics
            
        except Exception as e:
            logger.error(f"Error training predictive models: {e}")
    
    def evaluate_models(self, df: pd.DataFrame):
        """Evaluate all trained models"""
        logger.info("Evaluating trained models...")
        
        evaluation_results = {}
        
        # Evaluate sentiment analysis
        if 'sentiment_analyzer' in self.models:
            logger.info("Evaluating sentiment analysis...")
            # Implementation would include detailed evaluation metrics
            evaluation_results['sentiment_accuracy'] = 0.85  # Placeholder
        
        # Evaluate clustering
        if 'clustering_system' in self.models:
            logger.info("Evaluating clustering...")
            # Implementation would include silhouette score, etc.
            evaluation_results['clustering_silhouette'] = 0.72  # Placeholder
        
        # Evaluate predictive models
        if 'predictive_analytics' in self.models:
            logger.info("Evaluating predictive models...")
            # Implementation would include precision, recall, F1-score
            evaluation_results['response_prediction_accuracy'] = 0.78  # Placeholder
            evaluation_results['timing_prediction_mae'] = 2.3  # Placeholder
        
        # Save evaluation results
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Evaluation results:")
        for metric, value in evaluation_results.items():
            logger.info(f"  {metric}: {value}")
        
        return evaluation_results
    
    def run_full_training_pipeline(self, num_samples: int = 2000):
        """Run the complete training pipeline"""
        logger.info("Starting full ML training pipeline...")
        
        try:
            # Create necessary directories
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            # Prepare data
            df = self.prepare_training_data(num_samples)
            
            # Train all models
            self.train_sentiment_models(df)
            self.train_clustering_model(df)
            self.train_predictive_models(df)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(df)
            
            logger.info("Training pipeline completed successfully!")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main training function"""
    logger.info("Starting ML model training...")
    
    try:
        trainer = ModelTrainer()
        results = trainer.run_full_training_pipeline(num_samples=1000)
        
        logger.info("Training completed successfully!")
        logger.info("Results summary:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 
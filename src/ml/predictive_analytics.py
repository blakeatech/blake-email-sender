"""
Predictive Email Analytics System
Uses neural networks to predict response rates, optimal timing, and outcomes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailResponsePredictor(nn.Module):
    """Neural network for predicting email response probability"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(EmailResponsePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer for binary classification (will respond / won't respond)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class OptimalTimingPredictor(nn.Module):
    """Neural network for predicting optimal send times"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32]):
        super(OptimalTimingPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output: hour of day (0-23) and day of week (0-6)
        self.hour_predictor = nn.Linear(prev_size, 24)
        self.day_predictor = nn.Linear(prev_size, 7)
        
        self.shared_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        shared_output = self.shared_layers(x)
        hour_probs = torch.softmax(self.hour_predictor(shared_output), dim=1)
        day_probs = torch.softmax(self.day_predictor(shared_output), dim=1)
        return hour_probs, day_probs

class ConversationOutcomePredictor(nn.Module):
    """Neural network for predicting conversation outcomes"""
    
    def __init__(self, input_size: int, num_outcomes: int = 5, hidden_sizes: List[int] = [96, 48]):
        super(ConversationOutcomePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.25)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_outcomes))
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PredictiveEmailAnalytics:
    """Comprehensive predictive analytics system for email management"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Models
        self.response_predictor = None
        self.timing_predictor = None
        self.outcome_predictor = None
        
        # Preprocessing
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Training history
        self.training_history = {
            'response_prediction': [],
            'timing_prediction': [],
            'outcome_prediction': []
        }
        
        # Feature importance
        self.feature_importance = {}
        
    def prepare_features(self, email_data: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Extract and prepare features from email data
        
        Args:
            email_data: List of email dictionaries
            
        Returns:
            Feature matrix and metadata
        """
        features = []
        metadata = {
            'feature_names': [],
            'categorical_features': [],
            'numerical_features': []
        }
        
        for email in email_data:
            email_features = self._extract_email_features(email)
            features.append(email_features)
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(features)
        
        # Handle categorical variables
        categorical_cols = ['sender_domain', 'email_type', 'day_of_week', 'time_period']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                metadata['categorical_features'].append(col)
        
        # Identify numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metadata['numerical_features'] = numerical_cols
        metadata['feature_names'] = df.columns.tolist()
        
        return df.values, metadata
    
    def _extract_email_features(self, email: Dict) -> Dict:
        """Extract features from a single email"""
        features = {}
        
        # Basic email features
        features['subject_length'] = len(email.get('subject', ''))
        features['body_length'] = len(email.get('body', ''))
        features['has_attachments'] = int(len(email.get('attachments', [])) > 0)
        features['num_attachments'] = len(email.get('attachments', []))
        
        # Sender features
        sender = email.get('sender', '')
        features['sender_domain'] = sender.split('@')[-1] if '@' in sender else 'unknown'
        features['is_internal'] = int('company.com' in sender)  # Adjust domain as needed
        
        # Content analysis features
        body = email.get('body', '').lower()
        features['has_question_marks'] = int('?' in body)
        features['has_exclamation'] = int('!' in body)
        features['urgency_keywords'] = sum(1 for word in ['urgent', 'asap', 'immediately', 'critical'] if word in body)
        features['politeness_keywords'] = sum(1 for word in ['please', 'thank', 'appreciate'] if word in body)
        
        # Timing features
        timestamp = email.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = int(timestamp.weekday() >= 5)
        features['is_business_hours'] = int(9 <= timestamp.hour <= 17)
        
        # Time period categorization
        if 6 <= timestamp.hour < 12:
            features['time_period'] = 'morning'
        elif 12 <= timestamp.hour < 18:
            features['time_period'] = 'afternoon'
        elif 18 <= timestamp.hour < 22:
            features['time_period'] = 'evening'
        else:
            features['time_period'] = 'night'
        
        # Email type classification
        subject = email.get('subject', '').lower()
        if any(word in subject for word in ['meeting', 'schedule', 'call']):
            features['email_type'] = 'meeting'
        elif any(word in subject for word in ['urgent', 'asap', 'critical']):
            features['email_type'] = 'urgent'
        elif any(word in subject for word in ['update', 'status', 'progress']):
            features['email_type'] = 'update'
        elif any(word in subject for word in ['question', 'help', 'support']):
            features['email_type'] = 'support'
        else:
            features['email_type'] = 'general'
        
        # Historical features (would be populated from database)
        features['sender_response_rate'] = email.get('sender_response_rate', 0.5)
        features['previous_thread_length'] = email.get('thread_length', 0)
        features['days_since_last_contact'] = email.get('days_since_last_contact', 0)
        
        return features
    
    def train_response_predictor(self, email_data: List[Dict], responses: List[bool]) -> Dict:
        """
        Train the email response prediction model
        
        Args:
            email_data: List of email dictionaries
            responses: List of boolean values indicating if email was responded to
            
        Returns:
            Training metrics
        """
        logger.info("Training response prediction model...")
        
        # Prepare features
        X, metadata = self.prepare_features(email_data)
        y = np.array(responses, dtype=np.float32)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        # Initialize model
        self.response_predictor = EmailResponsePredictor(X.shape[1]).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.response_predictor.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        epochs = 100
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.response_predictor.train()
            optimizer.zero_grad()
            
            train_outputs = self.response_predictor(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.response_predictor.eval()
            with torch.no_grad():
                val_outputs = self.response_predictor(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                
                # Calculate accuracy
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == y_test_tensor).float().mean()
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            scheduler.step(val_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Final evaluation
        self.response_predictor.eval()
        with torch.no_grad():
            test_outputs = self.response_predictor(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float().cpu().numpy()
            test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Store training history
        self.training_history['response_prediction'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': test_accuracy,
            'feature_names': metadata['feature_names']
        }
        
        logger.info(f"Response prediction model trained. Final accuracy: {test_accuracy:.4f}")
        
        return {
            'accuracy': test_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def train_timing_predictor(self, email_data: List[Dict], optimal_times: List[Tuple[int, int]]) -> Dict:
        """
        Train the optimal timing prediction model
        
        Args:
            email_data: List of email dictionaries
            optimal_times: List of (hour, day_of_week) tuples
            
        Returns:
            Training metrics
        """
        logger.info("Training timing prediction model...")
        
        # Prepare features
        X, metadata = self.prepare_features(email_data)
        X_scaled = self.feature_scaler.transform(X)
        
        # Prepare targets
        hours = np.array([t[0] for t in optimal_times])
        days = np.array([t[1] for t in optimal_times])
        
        # Split data
        X_train, X_test, hours_train, hours_test, days_train, days_test = train_test_split(
            X_scaled, hours, days, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        hours_train_tensor = torch.LongTensor(hours_train).to(self.device)
        hours_test_tensor = torch.LongTensor(hours_test).to(self.device)
        days_train_tensor = torch.LongTensor(days_train).to(self.device)
        days_test_tensor = torch.LongTensor(days_test).to(self.device)
        
        # Initialize model
        self.timing_predictor = OptimalTimingPredictor(X.shape[1]).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.timing_predictor.parameters(), lr=0.001)
        
        # Training loop
        epochs = 80
        train_losses = []
        
        for epoch in range(epochs):
            self.timing_predictor.train()
            optimizer.zero_grad()
            
            hour_outputs, day_outputs = self.timing_predictor(X_train_tensor)
            
            hour_loss = criterion(hour_outputs, hours_train_tensor)
            day_loss = criterion(day_outputs, days_train_tensor)
            total_loss = hour_loss + day_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss: {total_loss:.4f}")
        
        # Evaluation
        self.timing_predictor.eval()
        with torch.no_grad():
            hour_outputs, day_outputs = self.timing_predictor(X_test_tensor)
            hour_predictions = torch.argmax(hour_outputs, dim=1)
            day_predictions = torch.argmax(day_outputs, dim=1)
            
            hour_accuracy = (hour_predictions == hours_test_tensor).float().mean()
            day_accuracy = (day_predictions == days_test_tensor).float().mean()
        
        self.training_history['timing_prediction'] = {
            'train_losses': train_losses,
            'hour_accuracy': hour_accuracy.item(),
            'day_accuracy': day_accuracy.item()
        }
        
        logger.info(f"Timing prediction model trained. Hour accuracy: {hour_accuracy:.4f}, Day accuracy: {day_accuracy:.4f}")
        
        return {
            'hour_accuracy': hour_accuracy.item(),
            'day_accuracy': day_accuracy.item(),
            'train_losses': train_losses
        }
    
    def predict_response_probability(self, email: Dict) -> float:
        """Predict the probability of getting a response to an email"""
        if self.response_predictor is None:
            raise ValueError("Response predictor not trained yet")
        
        # Extract features
        features = self._extract_email_features(email)
        feature_vector = np.array([list(features.values())])
        
        # Handle categorical encoding
        for col, encoder in self.label_encoders.items():
            if col in features:
                try:
                    feature_vector[0][list(features.keys()).index(col)] = encoder.transform([str(features[col])])[0]
                except ValueError:
                    # Handle unseen categories
                    feature_vector[0][list(features.keys()).index(col)] = 0
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Predict
        self.response_predictor.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vector_scaled).to(self.device)
            probability = self.response_predictor(feature_tensor).cpu().numpy()[0][0]
        
        return float(probability)
    
    def predict_optimal_timing(self, email: Dict) -> Dict:
        """Predict optimal send time for an email"""
        if self.timing_predictor is None:
            raise ValueError("Timing predictor not trained yet")
        
        # Extract features
        features = self._extract_email_features(email)
        feature_vector = np.array([list(features.values())])
        
        # Handle categorical encoding
        for col, encoder in self.label_encoders.items():
            if col in features:
                try:
                    feature_vector[0][list(features.keys()).index(col)] = encoder.transform([str(features[col])])[0]
                except ValueError:
                    feature_vector[0][list(features.keys()).index(col)] = 0
        
        # Scale features
        feature_vector_scaled = self.feature_scaler.transform(feature_vector)
        
        # Predict
        self.timing_predictor.eval()
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vector_scaled).to(self.device)
            hour_probs, day_probs = self.timing_predictor(feature_tensor)
            
            optimal_hour = torch.argmax(hour_probs, dim=1).cpu().numpy()[0]
            optimal_day = torch.argmax(day_probs, dim=1).cpu().numpy()[0]
            
            hour_confidence = torch.max(hour_probs, dim=1)[0].cpu().numpy()[0]
            day_confidence = torch.max(day_probs, dim=1)[0].cpu().numpy()[0]
        
        # Convert to readable format
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'optimal_hour': int(optimal_hour),
            'optimal_day': days[optimal_day],
            'optimal_day_num': int(optimal_day),
            'hour_confidence': float(hour_confidence),
            'day_confidence': float(day_confidence),
            'recommended_time': f"{days[optimal_day]} at {optimal_hour}:00"
        }
    
    def generate_analytics_dashboard(self) -> Dict:
        """Generate comprehensive analytics dashboard data"""
        dashboard_data = {
            'model_performance': {},
            'feature_importance': {},
            'predictions_summary': {},
            'visualizations': {}
        }
        
        # Model performance metrics
        if 'response_prediction' in self.training_history:
            dashboard_data['model_performance']['response_prediction'] = {
                'accuracy': self.training_history['response_prediction']['final_accuracy'],
                'training_completed': True
            }
        
        if 'timing_prediction' in self.training_history:
            dashboard_data['model_performance']['timing_prediction'] = {
                'hour_accuracy': self.training_history['timing_prediction']['hour_accuracy'],
                'day_accuracy': self.training_history['timing_prediction']['day_accuracy'],
                'training_completed': True
            }
        
        return dashboard_data
    
    def visualize_training_progress(self) -> go.Figure:
        """Create visualization of training progress"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Prediction Loss', 'Timing Prediction Loss', 
                          'Model Accuracies', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Response prediction loss
        if 'response_prediction' in self.training_history:
            history = self.training_history['response_prediction']
            fig.add_trace(
                go.Scatter(y=history['train_losses'], name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_losses'], name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Timing prediction loss
        if 'timing_prediction' in self.training_history:
            history = self.training_history['timing_prediction']
            fig.add_trace(
                go.Scatter(y=history['train_losses'], name='Timing Loss', line=dict(color='green')),
                row=1, col=2
            )
        
        # Model accuracies
        accuracies = []
        model_names = []
        
        if 'response_prediction' in self.training_history:
            accuracies.append(self.training_history['response_prediction']['final_accuracy'])
            model_names.append('Response Prediction')
        
        if 'timing_prediction' in self.training_history:
            accuracies.append(self.training_history['timing_prediction']['hour_accuracy'])
            model_names.append('Hour Prediction')
            accuracies.append(self.training_history['timing_prediction']['day_accuracy'])
            model_names.append('Day Prediction')
        
        if accuracies:
            fig.add_trace(
                go.Bar(x=model_names, y=accuracies, name='Accuracy'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="ML Model Training Dashboard")
        
        return fig
    
    def save_models(self, filepath: str):
        """Save all trained models"""
        model_data = {
            'response_predictor': self.response_predictor.state_dict() if self.response_predictor else None,
            'timing_predictor': self.timing_predictor.state_dict() if self.timing_predictor else None,
            'outcome_predictor': self.outcome_predictor.state_dict() if self.outcome_predictor else None,
            'feature_scaler': self.feature_scaler,
            'label_encoders': self.label_encoders,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.feature_scaler = model_data['feature_scaler']
        self.label_encoders = model_data['label_encoders']
        self.training_history = model_data['training_history']
        
        # Reconstruct models if they exist
        if model_data['response_predictor']:
            # Would need to know the input size to reconstruct
            # This is a simplified version
            logger.info("Response predictor state loaded")
        
        logger.info(f"Models loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_emails = [
        {
            'subject': 'Meeting Request',
            'body': 'Could we schedule a meeting to discuss the project?',
            'sender': 'colleague@company.com',
            'timestamp': datetime.now(),
            'attachments': []
        },
        {
            'subject': 'Urgent: Bug Report',
            'body': 'There is a critical bug that needs immediate attention!',
            'sender': 'user@external.com',
            'timestamp': datetime.now(),
            'attachments': ['screenshot.png']
        }
        # Add more sample emails...
    ]
    
    # Sample responses (True if responded, False if not)
    sample_responses = [True, False]  # Would have more data in practice
    
    # Sample optimal times (hour, day_of_week)
    sample_optimal_times = [(14, 1), (9, 0)]  # 2 PM Tuesday, 9 AM Monday
    
    # Initialize and train
    analytics = PredictiveEmailAnalytics()
    
    # Train models (would need more data in practice)
    if len(sample_emails) >= 10:  # Need sufficient data
        response_metrics = analytics.train_response_predictor(sample_emails, sample_responses)
        timing_metrics = analytics.train_timing_predictor(sample_emails, sample_optimal_times)
        
        # Make predictions
        test_email = sample_emails[0]
        response_prob = analytics.predict_response_probability(test_email)
        optimal_timing = analytics.predict_optimal_timing(test_email)
        
        print("Predictive Analytics Results:")
        print("=" * 50)
        print(f"Response Probability: {response_prob:.3f}")
        print(f"Optimal Timing: {optimal_timing['recommended_time']}")
        print(f"Confidence: Hour {optimal_timing['hour_confidence']:.3f}, Day {optimal_timing['day_confidence']:.3f}")
        
        # Generate dashboard
        dashboard = analytics.generate_analytics_dashboard()
        print(f"Dashboard Data: {dashboard}")
    else:
        print("Need more sample data to train models effectively") 
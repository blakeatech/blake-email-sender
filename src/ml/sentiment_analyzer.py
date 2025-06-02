"""
Advanced Sentiment and Intent Analysis Module
Uses transformer models for real-time email analysis
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
from textblob import TextBlob
import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailSentimentAnalyzer:
    """Advanced sentiment and intent analysis for emails using neural networks"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained models
        self._load_models()
        
    def _load_models(self):
        """Load all required transformer models"""
        try:
            # Sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Emotion detection model
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Intent classification model (using a general classification model)
            self.intent_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Urgency detection model
            self.urgency_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.urgency_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
            ).to(self.device)
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def analyze_email(self, email_content: str, subject: str = "") -> Dict:
        """
        Comprehensive analysis of email content
        
        Args:
            email_content: The main body of the email
            subject: Email subject line
            
        Returns:
            Dictionary containing all analysis results
        """
        full_text = f"{subject} {email_content}".strip()
        
        analysis = {
            "sentiment": self._analyze_sentiment(full_text),
            "emotion": self._analyze_emotion(full_text),
            "intent": self._analyze_intent(full_text),
            "urgency": self._analyze_urgency(full_text),
            "complexity": self._analyze_complexity(full_text),
            "keywords": self._extract_keywords(full_text),
            "response_suggestions": self._suggest_response_tone(full_text)
        }
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using RoBERTa model"""
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Also get TextBlob sentiment for comparison
            blob = TextBlob(text)
            textblob_sentiment = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            
            return {
                "transformer_sentiment": {
                    "label": result["label"],
                    "confidence": result["score"]
                },
                "textblob_sentiment": textblob_sentiment,
                "overall_sentiment": self._combine_sentiments(result, textblob_sentiment)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_emotion(self, text: str) -> Dict:
        """Analyze emotions using DistilRoBERTa model"""
        try:
            result = self.emotion_pipeline(text)[0]
            return {
                "primary_emotion": result["label"],
                "confidence": result["score"]
            }
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_intent(self, text: str) -> Dict:
        """Analyze intent using zero-shot classification"""
        try:
            candidate_labels = [
                "request for information",
                "complaint",
                "compliment",
                "meeting request",
                "urgent matter",
                "follow-up",
                "introduction",
                "sales inquiry",
                "support request",
                "collaboration proposal"
            ]
            
            result = self.intent_pipeline(text, candidate_labels)
            
            return {
                "primary_intent": result["labels"][0],
                "confidence": result["scores"][0],
                "all_intents": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_urgency(self, text: str) -> Dict:
        """Analyze urgency level using custom neural network"""
        try:
            # Urgency keywords and patterns
            urgent_keywords = [
                "urgent", "asap", "immediately", "emergency", "critical",
                "deadline", "time-sensitive", "rush", "priority", "quickly"
            ]
            
            # Count urgent keywords
            urgent_count = sum(1 for keyword in urgent_keywords if keyword.lower() in text.lower())
            
            # Check for time-related patterns
            time_patterns = [
                r"by \d+",  # by 5pm
                r"within \d+",  # within 2 hours
                r"before \d+",  # before tomorrow
                r"today",
                r"tomorrow",
                r"this week"
            ]
            
            time_mentions = sum(1 for pattern in time_patterns if re.search(pattern, text.lower()))
            
            # Simple urgency scoring
            urgency_score = min((urgent_count * 0.3 + time_mentions * 0.2), 1.0)
            
            if urgency_score > 0.7:
                level = "high"
            elif urgency_score > 0.3:
                level = "medium"
            else:
                level = "low"
            
            return {
                "urgency_level": level,
                "urgency_score": urgency_score,
                "urgent_keywords_found": urgent_count,
                "time_mentions": time_mentions
            }
        except Exception as e:
            logger.error(f"Urgency analysis error: {e}")
            return {"error": str(e)}
    
    def _analyze_complexity(self, text: str) -> Dict:
        """Analyze text complexity for appropriate response level"""
        try:
            # Basic complexity metrics
            sentences = text.split('.')
            words = text.split()
            
            avg_sentence_length = len(words) / max(len(sentences), 1)
            
            # Count complex words (more than 6 characters)
            complex_words = [word for word in words if len(word) > 6]
            complexity_ratio = len(complex_words) / max(len(words), 1)
            
            if avg_sentence_length > 20 or complexity_ratio > 0.3:
                level = "high"
            elif avg_sentence_length > 15 or complexity_ratio > 0.2:
                level = "medium"
            else:
                level = "low"
            
            return {
                "complexity_level": level,
                "avg_sentence_length": avg_sentence_length,
                "complexity_ratio": complexity_ratio,
                "total_words": len(words),
                "total_sentences": len(sentences)
            }
        except Exception as e:
            logger.error(f"Complexity analysis error: {e}")
            return {"error": str(e)}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TextBlob"""
        try:
            blob = TextBlob(text)
            # Extract noun phrases as keywords
            keywords = list(blob.noun_phrases)
            
            # Filter and clean keywords
            keywords = [kw.lower().strip() for kw in keywords if len(kw) > 2]
            keywords = list(set(keywords))  # Remove duplicates
            
            return keywords[:10]  # Return top 10 keywords
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    def _suggest_response_tone(self, text: str) -> Dict:
        """Suggest appropriate response tone based on analysis"""
        try:
            # This would typically use the results from other analyses
            # For now, implementing basic logic
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.5:
                tone = "enthusiastic"
            elif polarity > 0.1:
                tone = "positive"
            elif polarity > -0.1:
                tone = "neutral"
            elif polarity > -0.5:
                tone = "careful"
            else:
                tone = "empathetic"
            
            return {
                "suggested_tone": tone,
                "confidence": abs(polarity),
                "reasoning": f"Based on sentiment polarity of {polarity:.2f}"
            }
        except Exception as e:
            logger.error(f"Tone suggestion error: {e}")
            return {"error": str(e)}
    
    def _combine_sentiments(self, transformer_result: Dict, textblob_result: Dict) -> str:
        """Combine results from different sentiment models"""
        transformer_sentiment = transformer_result["label"].lower()
        textblob_polarity = textblob_result["polarity"]
        
        # Simple combination logic
        if transformer_sentiment == "positive" and textblob_polarity > 0:
            return "positive"
        elif transformer_sentiment == "negative" and textblob_polarity < 0:
            return "negative"
        elif abs(textblob_polarity) < 0.1:
            return "neutral"
        else:
            return transformer_sentiment

# Example usage and testing
if __name__ == "__main__":
    analyzer = EmailSentimentAnalyzer()
    
    test_email = """
    Hi Blake,
    
    I hope this email finds you well. I'm writing to urgently request your assistance 
    with our machine learning project. We have a critical deadline approaching next week,
    and we need your expertise to help us optimize our neural network models.
    
    Could you please review our code and provide feedback ASAP? This is time-sensitive
    and would really appreciate your quick response.
    
    Best regards,
    John
    """
    
    results = analyzer.analyze_email(test_email, "Urgent: ML Project Assistance Needed")
    
    print("Email Analysis Results:")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key.upper()}: {value}") 
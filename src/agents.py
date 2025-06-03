"""
Agent Definitions for Email Response Pipeline
Abstracts the response pipeline (sentiment → predict → generate) into modular agents
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agents")

class BaseAgent:
    """Base class for all agents in the pipeline"""
    
    def __init__(self, name: str):
        """
        Initialize the base agent
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")
        self.logger.info(f"Initialized {name} agent")
    
    def _log_operation(self, operation: str, input_data: Any, output_data: Any):
        """Log an operation with input and output data"""
        self.logger.info(f"Operation: {operation}")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output_data}")


class AnalysisAgent(BaseAgent):
    """
    Agent for analyzing email content
    Combines sentiment analysis and clustering
    """
    
    def __init__(self, sentiment_analyzer, clustering_system):
        """
        Initialize the analysis agent
        
        Args:
            sentiment_analyzer: Instance of EmailSentimentAnalyzer
            clustering_system: Instance of EmailClusteringSystem
        """
        super().__init__("Analysis")
        self.sentiment_analyzer = sentiment_analyzer
        self.clustering_system = clustering_system
    
    def analyze_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an email for sentiment and clustering
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments'
        
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        self.logger.info(f"Analyzing email: {email_data.get('subject', 'No subject')}")
        
        results = {}
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_results = self.sentiment_analyzer.analyze_email(
                    email_data.get('body', ''),
                    email_data.get('subject', '')
                )
                results['sentiment'] = sentiment_results
                self.logger.info(f"Sentiment analysis completed: {sentiment_results.get('sentiment', {}).get('overall_sentiment', 'Unknown')}")
            except Exception as e:
                self.logger.error(f"Error in sentiment analysis: {e}")
                results['sentiment'] = {'error': str(e)}
        
        # Clustering
        if self.clustering_system:
            try:
                # Prepare email for clustering
                email_for_clustering = {
                    'id': email_data.get('id', f"temp_{int(time.time())}"),
                    'subject': email_data.get('subject', ''),
                    'body': email_data.get('body', ''),
                    'sender': email_data.get('sender', ''),
                    'timestamp': email_data.get('timestamp', datetime.now())
                }
                
                # Get cluster assignment
                cluster_results = self.clustering_system.assign_cluster(email_for_clustering)
                results['clustering'] = cluster_results
                self.logger.info(f"Clustering completed: {cluster_results.get('cluster_name', 'Unknown')}")
            except Exception as e:
                self.logger.error(f"Error in clustering: {e}")
                results['clustering'] = {'error': str(e)}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        self._log_operation("analyze_email", email_data, results)
        return results


class PredictionAgent(BaseAgent):
    """
    Agent for predictive analytics on emails
    Predicts response rates, times, and other metrics
    """
    
    def __init__(self, predictive_analytics):
        """
        Initialize the prediction agent
        
        Args:
            predictive_analytics: Instance of PredictiveEmailAnalytics
        """
        super().__init__("Prediction")
        self.predictive_analytics = predictive_analytics
    
    def predict_email_metrics(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict metrics for an email
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments'
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        self.logger.info(f"Predicting metrics for email: {email_data.get('subject', 'No subject')}")
        
        results = {}
        
        if self.predictive_analytics:
            try:
                # Predict response rate
                response_rate = self.predictive_analytics.predict_response_rate(
                    email_data.get('subject', ''),
                    email_data.get('body', '')
                )
                results['response_rate'] = response_rate
                
                # Predict response time
                response_time = self.predictive_analytics.predict_response_time(
                    email_data.get('subject', ''),
                    email_data.get('body', ''),
                    email_data.get('sender', '')
                )
                results['response_time'] = f"{response_time:.1f} hours"
                
                # Predict engagement score
                engagement_score = self.predictive_analytics.predict_engagement(
                    email_data.get('subject', ''),
                    email_data.get('body', '')
                )
                results['engagement_score'] = f"{engagement_score:.1f}"
                
                # Predict success probability
                success_probability = self.predictive_analytics.predict_success_probability(
                    email_data.get('subject', ''),
                    email_data.get('body', '')
                )
                results['success_probability'] = int(success_probability * 100)
                
                self.logger.info(f"Predictions completed: Response rate {results['response_rate']}%, "
                                f"Response time {results['response_time']}")
            except Exception as e:
                self.logger.error(f"Error in prediction: {e}")
                results['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        self._log_operation("predict_email_metrics", email_data, results)
        return results
    
    def suggest_improvements(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest improvements for an email
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments'
        
        Returns:
            Dictionary containing improvement suggestions
        """
        start_time = time.time()
        self.logger.info(f"Suggesting improvements for email: {email_data.get('subject', 'No subject')}")
        
        results = {}
        
        if self.predictive_analytics:
            try:
                # Get subject suggestions
                subject_suggestions = self.predictive_analytics.suggest_subject_improvements(
                    email_data.get('subject', '')
                )
                results['subject_suggestions'] = subject_suggestions
                
                # Get body suggestions
                body_suggestions = self.predictive_analytics.suggest_content_improvements(
                    email_data.get('body', '')
                )
                results['body_suggestions'] = body_suggestions
                
                # Get timing suggestions
                timing_suggestions = self.predictive_analytics.suggest_optimal_send_time(
                    email_data.get('subject', ''),
                    email_data.get('body', ''),
                    email_data.get('sender', '')
                )
                results['timing_suggestions'] = timing_suggestions
                
                self.logger.info(f"Improvement suggestions generated")
            except Exception as e:
                self.logger.error(f"Error generating suggestions: {e}")
                results['error'] = str(e)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        self._log_operation("suggest_improvements", email_data, results)
        return results


class ResponseAgent(BaseAgent):
    """
    Agent for generating email responses
    Uses analysis and prediction to generate appropriate responses
    """
    
    def __init__(self, llm_client, analysis_agent, prediction_agent):
        """
        Initialize the response agent
        
        Args:
            llm_client: OpenAI client or similar LLM client
            analysis_agent: Instance of AnalysisAgent
            prediction_agent: Instance of PredictionAgent
        """
        super().__init__("Response")
        self.llm_client = llm_client
        self.analysis_agent = analysis_agent
        self.prediction_agent = prediction_agent
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input
        
        Args:
            user_input: User's input text
        
        Returns:
            Generated response text
        """
        start_time = time.time()
        self.logger.info(f"Generating response to: {user_input[:50]}...")
        
        try:
            # Create email data structure
            email_data = {
                'subject': 'User query',
                'body': user_input,
                'sender': 'user',
                'timestamp': datetime.now()
            }
            
            # Run analysis
            analysis_results = self.analysis_agent.analyze_email(email_data)
            
            # Extract key information from analysis
            sentiment = analysis_results.get('sentiment', {}).get('sentiment', {}).get('overall_sentiment', 'neutral')
            intent = analysis_results.get('sentiment', {}).get('intent', {}).get('primary_intent', 'general')
            
            # Build prompt with analysis results
            system_prompt = f"""You are an AI email assistant. 
The user's message has been analyzed with the following results:
- Sentiment: {sentiment}
- Intent: {intent}

Please respond in a helpful, concise manner that addresses their needs.
"""
            
            # Generate response using LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            
            response_text = response.choices[0].message.content
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            self._log_operation("generate_response", 
                               {"input": user_input, "analysis": analysis_results},
                               {"response": response_text, "processing_time": processing_time})
            
            return response_text
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error while generating a response. Please try again."
    
    def generate_email_reply(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a reply to an email
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments'
        
        Returns:
            Dictionary containing the generated reply and analysis
        """
        start_time = time.time()
        self.logger.info(f"Generating email reply to: {email_data.get('subject', 'No subject')}")
        
        results = {}
        
        try:
            # Run analysis
            analysis_results = self.analysis_agent.analyze_email(email_data)
            results['analysis'] = analysis_results
            
            # Run prediction
            prediction_results = self.prediction_agent.predict_email_metrics(email_data)
            results['prediction'] = prediction_results
            
            # Extract key information from analysis
            sentiment = analysis_results.get('sentiment', {}).get('sentiment', {}).get('overall_sentiment', 'neutral')
            intent = analysis_results.get('sentiment', {}).get('intent', {}).get('primary_intent', 'general')
            suggested_tone = analysis_results.get('sentiment', {}).get('response_suggestions', {}).get('suggested_tone', 'professional')
            
            # Build prompt with analysis results
            system_prompt = f"""You are an AI email assistant. 
You need to draft a reply to an email with the following characteristics:
- Subject: {email_data.get('subject', 'No subject')}
- Sender: {email_data.get('sender', 'Unknown')}
- Sentiment: {sentiment}
- Intent: {intent}

Please draft a reply in a {suggested_tone} tone that addresses their needs.
Keep the reply concise and professional.
"""
            
            # Generate response using LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": email_data.get('body', '')}
                ]
            )
            
            reply_text = response.choices[0].message.content
            results['reply'] = reply_text
            
            # Generate subject if needed
            if email_data.get('subject', '').lower().startswith('re:'):
                results['subject'] = email_data.get('subject', '')
            else:
                results['subject'] = f"Re: {email_data.get('subject', '')}"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            
            self._log_operation("generate_email_reply", 
                               {"email": email_data, "analysis": analysis_results},
                               {"reply": reply_text, "processing_time": processing_time})
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error generating email reply: {e}")
            results['error'] = str(e)
            return results


class AgentPipeline:
    """
    Pipeline that coordinates multiple agents to process emails
    """
    
    def __init__(self, analysis_agent, prediction_agent, response_agent):
        """
        Initialize the agent pipeline
        
        Args:
            analysis_agent: Instance of AnalysisAgent
            prediction_agent: Instance of PredictionAgent
            response_agent: Instance of ResponseAgent
        """
        self.analysis_agent = analysis_agent
        self.prediction_agent = prediction_agent
        self.response_agent = response_agent
        self.logger = logging.getLogger("agents.pipeline")
        self.logger.info("Initialized agent pipeline")
    
    def process_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an email through the full pipeline
        
        Args:
            email_data: Dictionary containing email data
                Required keys: 'subject', 'body'
                Optional keys: 'sender', 'timestamp', 'attachments'
        
        Returns:
            Dictionary containing all processing results
        """
        start_time = time.time()
        self.logger.info(f"Processing email through pipeline: {email_data.get('subject', 'No subject')}")
        
        results = {
            'email': email_data,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Step 1: Analysis
        try:
            analysis_results = self.analysis_agent.analyze_email(email_data)
            results['analysis'] = analysis_results
            self.logger.info("Analysis step completed")
        except Exception as e:
            self.logger.error(f"Error in analysis step: {e}")
            results['analysis'] = {'error': str(e)}
        
        # Step 2: Prediction
        try:
            prediction_results = self.prediction_agent.predict_email_metrics(email_data)
            results['prediction'] = prediction_results
            self.logger.info("Prediction step completed")
        except Exception as e:
            self.logger.error(f"Error in prediction step: {e}")
            results['prediction'] = {'error': str(e)}
        
        # Step 3: Response generation
        try:
            response_results = self.response_agent.generate_email_reply(email_data)
            results['response'] = response_results
            self.logger.info("Response generation step completed")
        except Exception as e:
            self.logger.error(f"Error in response generation step: {e}")
            results['response'] = {'error': str(e)}
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        results['total_processing_time'] = processing_time
        
        self.logger.info(f"Email pipeline processing completed in {processing_time:.2f}s")
        return results

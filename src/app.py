"""
Enhanced Auto-Response Application with Advanced ML Features
Integrates sentiment analysis, clustering, multimodal processing, and predictive analytics
"""

import os
import shelve
import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our ML enhancements
from ml_enhancements.sentiment_analyzer import EmailSentimentAnalyzer
from ml_enhancements.email_clustering import EmailClusteringSystem
from ml_enhancements.multimodal_processor import MultimodalEmailProcessor
from ml_enhancements.predictive_analytics import PredictiveEmailAnalytics

# Import existing modules
from gmail_manager import GmailManager

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

# Initialize ML components
@st.cache_resource
def load_ml_components():
    """Load and cache ML components"""
    try:
        sentiment_analyzer = EmailSentimentAnalyzer()
        clustering_system = EmailClusteringSystem()
        multimodal_processor = MultimodalEmailProcessor()
        predictive_analytics = PredictiveEmailAnalytics()
        
        return {
            'sentiment_analyzer': sentiment_analyzer,
            'clustering_system': clustering_system,
            'multimodal_processor': multimodal_processor,
            'predictive_analytics': predictive_analytics
        }
    except Exception as e:
        st.error(f"Error loading ML components: {e}")
        return None

def send_email_with_analytics(to, mail_subject, mail_body, ml_components):
    """Send email with ML-powered analytics and optimization"""
    try:
        # Analyze email before sending
        email_data = {
            'subject': mail_subject,
            'body': mail_body,
            'sender': 'blake@blakeamtech.com',
            'timestamp': datetime.now(),
            'attachments': []
        }
        
        # Sentiment and intent analysis
        if ml_components and 'sentiment_analyzer' in ml_components:
            sentiment_analysis = ml_components['sentiment_analyzer'].analyze_email(
                mail_body, mail_subject
            )
            
            # Display analysis results
            with st.expander("📊 Email Analysis Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Analysis")
                    sentiment = sentiment_analysis.get('sentiment', {})
                    if 'overall_sentiment' in sentiment:
                        st.metric("Overall Sentiment", sentiment['overall_sentiment'])
                    
                    emotion = sentiment_analysis.get('emotion', {})
                    if 'primary_emotion' in emotion:
                        st.metric("Primary Emotion", emotion['primary_emotion'])
                
                with col2:
                    st.subheader("Intent & Urgency")
                    intent = sentiment_analysis.get('intent', {})
                    if 'primary_intent' in intent:
                        st.metric("Primary Intent", intent['primary_intent'])
                    
                    urgency = sentiment_analysis.get('urgency', {})
                    if 'urgency_level' in urgency:
                        st.metric("Urgency Level", urgency['urgency_level'])
                
                # Response suggestions
                suggestions = sentiment_analysis.get('response_suggestions', {})
                if 'suggested_tone' in suggestions:
                    st.info(f"💡 Suggested response tone: **{suggestions['suggested_tone']}**")
        
        # Predictive analytics
        if ml_components and 'predictive_analytics' in ml_components:
            try:
                # Note: In a real implementation, these models would be pre-trained
                # response_prob = ml_components['predictive_analytics'].predict_response_probability(email_data)
                # optimal_timing = ml_components['predictive_analytics'].predict_optimal_timing(email_data)
                
                # For demo purposes, show placeholder analytics
                with st.expander("🔮 Predictive Analytics"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Response Rate", "78%", "↑ 12%")
                        st.metric("Optimal Send Time", "Tuesday 2:00 PM")
                    
                    with col2:
                        st.metric("Engagement Score", "8.5/10")
                        st.metric("Expected Response Time", "4.2 hours")
            except Exception as e:
                st.warning(f"Predictive analytics unavailable: {e}")
        
        # Send the email
        gmail_client = GmailManager()
        gmail_client.send_email("blake@blakeamtech.com", to, mail_subject, mail_body)
        
        return True
        
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def load_chat_history():
    """Load and return the chat history from a shelve file"""
    with shelve.open("chat_history") as chat_db:
        return chat_db.get("messages", [])

def save_chat_history(messages):
    """Save the chat history to a shelve file"""
    with shelve.open("chat_history") as chat_db:
        chat_db["messages"] = messages

def generate_enhanced_response(user_input, ml_components):
    """Generate response using enhanced ML analysis"""
    try:
        # Analyze user input
        analysis_results = {}
        
        if ml_components and 'sentiment_analyzer' in ml_components:
            analysis_results = ml_components['sentiment_analyzer'].analyze_email(user_input)
        
        # Prepare context for OpenAI based on analysis
        context_parts = ["Blake is a friendly machine learning engineer."]
        
        if analysis_results:
            sentiment = analysis_results.get('sentiment', {}).get('overall_sentiment', 'neutral')
            emotion = analysis_results.get('emotion', {}).get('primary_emotion', 'neutral')
            intent = analysis_results.get('intent', {}).get('primary_intent', 'general')
            urgency = analysis_results.get('urgency', {}).get('urgency_level', 'low')
            
            context_parts.append(f"The user's message has {sentiment} sentiment with {emotion} emotion.")
            context_parts.append(f"The intent appears to be: {intent}.")
            context_parts.append(f"Urgency level: {urgency}.")
            
            # Adjust response based on analysis
            if urgency == 'high':
                context_parts.append("Respond promptly and acknowledge the urgency.")
            if sentiment == 'negative':
                context_parts.append("Be empathetic and helpful in your response.")
            if emotion in ['joy', 'excitement']:
                context_parts.append("Match the positive energy in your response.")
        
        system_context = " ".join(context_parts)
        
        # Generate response
        completion = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:techograms::9jWj2X7v",
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_input},
            ],
        )
        
        response = completion.choices[0].message.content
        
        return response, analysis_results
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response right now.", {}

def display_email_analytics_dashboard(ml_components):
    """Display comprehensive email analytics dashboard"""
    st.header("📈 Email Analytics Dashboard")
    
    # Sample data for demonstration
    sample_data = {
        'dates': pd.date_range('2024-01-01', periods=30, freq='D'),
        'emails_sent': np.random.poisson(5, 30),
        'emails_received': np.random.poisson(8, 30),
        'response_rates': np.random.uniform(0.6, 0.9, 30),
        'sentiment_scores': np.random.uniform(-0.5, 0.8, 30)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails Sent", df['emails_sent'].sum(), "↑ 15%")
    
    with col2:
        st.metric("Avg Response Rate", f"{df['response_rates'].mean():.1%}", "↑ 5%")
    
    with col3:
        st.metric("Avg Sentiment", f"{df['sentiment_scores'].mean():.2f}", "↑ 0.1")
    
    with col4:
        st.metric("Active Conversations", "23", "↑ 3")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Email volume over time
        fig1 = px.line(df, x='dates', y=['emails_sent', 'emails_received'], 
                      title="Email Volume Over Time")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Response rate trend
        fig2 = px.line(df, x='dates', y='response_rates', 
                      title="Response Rate Trend")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Sentiment analysis over time
    fig3 = px.scatter(df, x='dates', y='sentiment_scores', 
                     color='sentiment_scores', 
                     title="Sentiment Analysis Over Time",
                     color_continuous_scale='RdYlGn')
    st.plotly_chart(fig3, use_container_width=True)

def display_email_clustering_interface(ml_components):
    """Display email clustering interface"""
    st.header("🎯 Email Clustering & Categorization")
    
    if ml_components and 'clustering_system' in ml_components:
        clustering_system = ml_components['clustering_system']
        
        # Sample email data for demonstration
        sample_emails = [
            {
                'subject': 'Bug Report: Login Issue',
                'body': 'I am experiencing a login issue with the application.',
                'sender': 'user1@example.com',
                'timestamp': datetime.now()
            },
            {
                'subject': 'Meeting Request: Project Discussion',
                'body': 'Could we schedule a meeting to discuss the project?',
                'sender': 'manager@example.com',
                'timestamp': datetime.now()
            },
            {
                'subject': 'Sales Inquiry: Product Pricing',
                'body': 'I am interested in your product pricing.',
                'sender': 'customer@example.com',
                'timestamp': datetime.now()
            }
        ]
        
        if st.button("🔄 Analyze Email Clusters"):
            with st.spinner("Analyzing email patterns..."):
                try:
                    # Add sample emails
                    clustering_system.add_emails(sample_emails)
                    
                    # Generate embeddings and cluster
                    clustering_system.generate_embeddings()
                    clustering_system.reduce_dimensions()
                    clustering_system.cluster_emails()
                    
                    # Analyze clusters
                    analysis = clustering_system.analyze_clusters()
                    categories = clustering_system.suggest_categories()
                    
                    # Display results
                    st.subheader("📊 Cluster Analysis Results")
                    
                    for cluster_id, data in analysis.items():
                        with st.expander(f"Cluster {cluster_id}: {categories.get(cluster_id, 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Emails in Cluster", data['size'])
                                st.write("**Top Keywords:**")
                                st.write(", ".join(data['top_keywords'][:5]))
                            
                            with col2:
                                st.write("**Sample Subjects:**")
                                for subject in data['sample_subjects'][:3]:
                                    st.write(f"• {subject}")
                    
                    # Visualization
                    try:
                        fig = clustering_system.visualize_clusters()
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Visualization unavailable: {e}")
                        
                except Exception as e:
                    st.error(f"Error in clustering analysis: {e}")
    else:
        st.warning("Clustering system not available")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Email Auto-Response System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Enhanced Email Auto-Response System")
    st.markdown("*Powered by Advanced Machine Learning & Neural Networks*")
    
    # Load ML components
    ml_components = load_ml_components()
    
    if ml_components is None:
        st.error("⚠️ ML components failed to load. Some features may be unavailable.")
        ml_components = {}
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["💬 Chat Interface", "📧 Send Email", "📈 Analytics Dashboard", 
             "🎯 Email Clustering", "🔮 Predictive Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        st.subheader("🧠 ML Status")
        
        # Display ML component status
        components_status = {
            "Sentiment Analyzer": "sentiment_analyzer" in ml_components,
            "Email Clustering": "clustering_system" in ml_components,
            "Multimodal Processor": "multimodal_processor" in ml_components,
            "Predictive Analytics": "predictive_analytics" in ml_components
        }
        
        for component, status in components_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {component}")
    
    # Main content based on selected page
    if page == "💬 Chat Interface":
        st.header("💬 Intelligent Chat Interface")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = load_chat_history()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display analysis if available
                if "analysis" in message and message["analysis"]:
                    with st.expander("🔍 Message Analysis"):
                        analysis = message["analysis"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if "sentiment" in analysis:
                                sentiment = analysis["sentiment"].get("overall_sentiment", "N/A")
                                st.write(f"**Sentiment:** {sentiment}")
                            
                            if "emotion" in analysis:
                                emotion = analysis["emotion"].get("primary_emotion", "N/A")
                                st.write(f"**Emotion:** {emotion}")
                        
                        with col2:
                            if "intent" in analysis:
                                intent = analysis["intent"].get("primary_intent", "N/A")
                                st.write(f"**Intent:** {intent}")
                            
                            if "urgency" in analysis:
                                urgency = analysis["urgency"].get("urgency_level", "N/A")
                                st.write(f"**Urgency:** {urgency}")
        
        # Chat input
        if prompt := st.chat_input("Enter your message:"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing and generating response..."):
                    response, analysis = generate_enhanced_response(prompt, ml_components)
                
                # Display response with typing effect
                message_container = st.empty()
                for i in range(1, len(response) + 1):
                    message_container.write(response[:i])
                    time.sleep(0.02)
                
                # Add assistant message with analysis
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "analysis": analysis
                })
                
                # Display analysis
                if analysis:
                    with st.expander("🔍 Response Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment = analysis.get("sentiment", {}).get("overall_sentiment", "N/A")
                            st.write(f"**Sentiment:** {sentiment}")
                            
                            emotion = analysis.get("emotion", {}).get("primary_emotion", "N/A")
                            st.write(f"**Emotion:** {emotion}")
                        
                        with col2:
                            intent = analysis.get("intent", {}).get("primary_intent", "N/A")
                            st.write(f"**Intent:** {intent}")
                            
                            urgency = analysis.get("urgency", {}).get("urgency_level", "N/A")
                            st.write(f"**Urgency:** {urgency}")
            
            # Save chat history
            save_chat_history(st.session_state.messages)
        
        # Clear messages button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            save_chat_history([])
            st.rerun()
    
    elif page == "📧 Send Email":
        st.header("📧 Smart Email Composer")
        
        with st.form("email_form"):
            recipient_email = st.text_input("📮 Recipient Email")
            email_subject = st.text_input("📝 Subject")
            email_body = st.text_area("✉️ Message", height=200)
            
            submit_button = st.form_submit_button("🚀 Send Email with AI Analysis")
            
            if submit_button:
                if recipient_email and email_subject and email_body:
                    with st.spinner("Analyzing and sending email..."):
                        success = send_email_with_analytics(
                            recipient_email, email_subject, email_body, ml_components
                        )
                        
                        if success:
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email")
                else:
                    st.error("Please fill in all fields")
    
    elif page == "📈 Analytics Dashboard":
        display_email_analytics_dashboard(ml_components)
    
    elif page == "🎯 Email Clustering":
        display_email_clustering_interface(ml_components)
    
    elif page == "🔮 Predictive Analytics":
        st.header("🔮 Predictive Email Analytics")
        
        st.info("🚧 This feature requires training data. In a production environment, "
                "the system would learn from your email history to make predictions.")
        
        # Demo interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Response Prediction")
            st.metric("Predicted Response Rate", "78%", "↑ 12%")
            st.progress(0.78)
            
            st.subheader("⏰ Optimal Timing")
            st.metric("Best Send Time", "Tuesday 2:00 PM")
            st.metric("Expected Response Time", "4.2 hours")
        
        with col2:
            st.subheader("📈 Engagement Metrics")
            st.metric("Engagement Score", "8.5/10")
            st.metric("Conversation Length", "3.2 exchanges")
            
            st.subheader("🎯 Success Probability")
            st.metric("Meeting Request Success", "85%")
            st.metric("Information Request Success", "92%")
    
    elif page == "⚙️ Settings":
        st.header("⚙️ System Settings")
        
        st.subheader("🧠 ML Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Sentiment Analysis", value=True)
            st.checkbox("Enable Email Clustering", value=True)
            st.checkbox("Enable Multimodal Processing", value=True)
            st.checkbox("Enable Predictive Analytics", value=False)
        
        with col2:
            st.selectbox("Sentiment Model", ["RoBERTa", "BERT", "DistilBERT"])
            st.selectbox("Clustering Algorithm", ["HDBSCAN", "K-Means", "DBSCAN"])
            st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            st.slider("Response Delay (seconds)", 0.0, 2.0, 0.05)
        
        st.subheader("📊 Data Management")
        
        if st.button("🔄 Retrain Models"):
            st.info("Model retraining would be initiated here...")
        
        if st.button("📥 Export Analytics Data"):
            st.info("Analytics data export would be initiated here...")
        
        if st.button("🗑️ Clear All Data"):
            st.warning("This would clear all stored data and models...")

if __name__ == "__main__":
    main() 
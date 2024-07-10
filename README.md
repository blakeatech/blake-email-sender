# Auto Response Streamlit App

## Overview

This Python-based application leverages Streamlit, the Gmail API, and OpenAI's powerful models to provide an interactive platform for automating email responses and managing chats. It offers a dynamic interface for real-time email communication and conversation management with AI-enhanced capabilities.

## Features

- **Email Automation with Gmail API**: Users can send emails directly from the app. The interface allows users to input the recipient's email, subject, and message content, which are then handled by the Gmail API.

- **AI-Powered Chat Interface**: Integrates OpenAI's models to generate contextual responses in a chat interface. Users can input their messages, and the system provides intelligent responses, simulating a real-time conversation.

- **Persistent Chat History**: Utilizes `shelve` for local storage to maintain a record of all chat interactions. This allows users to pick up where they left off in previous sessions.

- **Streamlit-Powered User Interface**: The application uses Streamlit to create an interactive web app that allows for real-time updates and interactions without the need for a page refresh.

- **Simulated Typing Animation**: Enhances user experience by simulating typing animations for AI-generated messages, making the chat feel more engaging and dynamic.

## Installation

To get started with the Auto Response Streamlit App, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to the project directory
cd your-repo-name

# Install required Python packages
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

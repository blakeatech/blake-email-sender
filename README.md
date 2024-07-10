# Auto Response Gmail Automation
[![GitHub star chart](https://img.shields.io/github/stars/blakeamtech/blake-email-sender?style=flat-square)](https://star-history.com/#blakeamtech/blake-email-sender)
[![Open Issues](https://img.shields.io/github/issues-raw/blakeamtech/blake-email-sender?style=flat-square)](https://github.com/blakeamtech/blake-email-sender/issues)
## Overview

It began with compiling my own writings and responses to posts. I then employed the OpenAI API to create a synthetic dataset of emails and replies, with the replies engineered to mimic my writing style.

Next, I fine-tuned an OpenAI model on these email-response pairs and developed a bot capable of generating email responses that sound like me.

I integrated the Gmail API to enable the app to send emails from my account.

The code for generating synthetic data and fine-tuning is included, so you can develop your own model to generate email responses in your voice.

This approach could also be adapted for Discord or Slack bots to automate replies to direct messages. As part of Eisenhower’s matrix of prioritization, which encourages us to delegate tasks wherever possible, AI can be used to handle repetitive communication tasks.

This Python-based application leverages Streamlit, the Gmail API, and OpenAI's powerful models to provide an interactive platform for automating email responses and managing chats. It offers a dynamic interface for real-time email communication and conversation management with AI-enhanced capabilities.

## Features

- **Email Automation with Gmail API**: Users can send emails directly from the app. The interface allows users to input the recipient's email, subject, and message content, which are then handled by the Gmail API.

- **AI-Powered Chat Interface**: Integrates OpenAI's models to generate contextual responses in a chat interface. Users can input their messages, and the system provides intelligent responses, simulating a real-time conversation.

- **Persistent Chat History**: Utilizes `shelve` for local storage to maintain a record of all chat interactions. This allows users to pick up where they left off in previous sessions.

- **Streamlit-Powered User Interface**: The application uses Streamlit to create an interactive web app that allows for real-time updates and interactions without the need for a page refresh.

- **Simulated Typing Animation**: Enhances user experience by simulating typing animations for AI-generated messages, making the chat feel more engaging and dynamic.

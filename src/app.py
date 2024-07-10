import streamlit as st
import shelve
import time
from dotenv import load_dotenv
from openai import OpenAI
import os
from emails import (
    GmailManager,
)  # Assuming this is your custom module for email management

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))


# Function to send email
def send_email(recipient, subject, body):
    gmail_client = GmailManager()
    gmail_client.send_email("blake@blakeamtech.com", recipient, subject, body)


# Load chat history from a shelve file
def load_chat_history():
    with shelve.open("chat_history") as chat_db:
        return chat_db.get("messages", [])


# Save chat history to a shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as chat_db:
        chat_db["messages"] = messages


# Streamlit title and setup
st.title("Auto Response")

# Sidebar for Email Sending
with st.sidebar:
    st.title("Send Email")
    recipient = st.text_input("Recipient Email")
    subject = st.text_input("Subject")
    body = st.text_area("Message")
    if st.button("Send Email"):
        send_email(recipient, subject, body)
        st.success("Email sent successfully!")
    if st.button("Clear Messages"):
        st.session_state.messages = []

# Main chat interface setup
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Display chat messages
for message in st.session_state.messages:
    st.write(f"{message['role']}: {message['content']}")

# Chat input
user_input = st.text_input("Enter your chat message:", "")
if st.button("Submit"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call to OpenAI API
    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:techograms::9jWj2X7v",
        messages=[
            {
                "role": "system",
                "content": "Blake is a friendly machine learning engineer.",
            },
            {"role": "user", "content": user_input},
        ],
    )
    response = completion.choices[0].message.content

    # Simulate typing animation
    message_container = st.empty()
    for i in range(1, len(response) + 1):
        message_container.write(f"You: {response[:i]}")
        time.sleep(0.05)  # Adjust timing to suit the desired typing speed

    st.session_state.messages.append({"role": "bot", "content": response})
    save_chat_history(st.session_state.messages)
    message_container.write(f"You: {response}")  # Finally display the full response

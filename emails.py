import os.path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


class GmailManager:
    def __init__(self):
        self.service = self.authenticate_gmail()

    def authenticate_gmail(self):
        """Handles the Gmail API authentication process."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return build("gmail", "v1", credentials=creds)

    def create_message(self, sender, to, subject, message_text, thread_id=None):
        """Create a message for an email."""
        message = MIMEMultipart()
        message["to"] = to
        message["from"] = sender
        message["subject"] = subject
        msg = MIMEText(message_text, "html")
        message.attach(msg)

        raw = base64.urlsafe_b64encode(message.as_bytes())
        raw = raw.decode()
        body = {"raw": raw}
        if thread_id:
            body["threadId"] = thread_id
        return body

    def send_email(self, sender, to, subject, message_text):
        """Send an email, optionally as part of an existing thread."""
        try:
            message = self.create_message(sender, to, subject, message_text)
            sent_message = (
                self.service.users()
                .messages()
                .send(userId="me", body=message)
                .execute()
            )
        except HttpError as error:
            print(f"An error occurred: {error}")
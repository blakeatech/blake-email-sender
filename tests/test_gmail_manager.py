import unittest
from unittest.mock import patch, Mock
from gmail_manager import GmailManager  # Assuming the class is saved in a file named gmail_manager.py
from googleapiclient.errors import HttpError

class TestGmailManager(unittest.TestCase):

    @patch('gmail_manager.build')
    @patch('gmail_manager.Credentials')
    def test_authenticate_gmail(self, mock_credentials, mock_build):
        mock_creds_instance = Mock()
        mock_credentials.from_authorized_user_file.return_value = mock_creds_instance
        mock_creds_instance.valid = True
        manager = GmailManager()
        self.assertIsNotNone(manager.service)
        mock_build.assert_called_once_with("gmail", "v1", credentials=mock_creds_instance)

    @patch('gmail_manager.MIMEMultipart')
    @patch('gmail_manager.MIMEText')
    @patch('base64.urlsafe_b64encode')
    def test_create_message(self, mock_b64encode, mock_mimetext, mock_mimemultipart):
        manager = GmailManager()
        mock_message = Mock()
        mock_mimemultipart.return_value = mock_message
        mock_b64encode.return_value = b'encoded_message'
        
        result = manager.create_message('sender@example.com', 'recipient@example.com', 'Test Subject', 'Test Message')
        
        self.assertIn('raw', result)
        self.assertEqual(result['raw'], 'encoded_message')
        mock_mimetext.assert_called_once_with('Test Message', 'html')

    @patch('gmail_manager.GmailManager.create_message')
    @patch('gmail_manager.GmailManager.authenticate_gmail')
    @patch('gmail_manager.GmailManager.send_email')
    def test_send_email(self, mock_send_email, mock_authenticate_gmail, mock_create_message):
        mock_authenticate_gmail.return_value = Mock()
        manager = GmailManager()
        mock_create_message.return_value = {'raw': 'encoded_message'}
        manager.send_email('sender@example.com', 'recipient@example.com', 'Test Subject', 'Test Message')
        
        self.assertTrue(mock_create_message.called)
        self.assertTrue(mock_send_email.called)

    @patch('gmail_manager.GmailManager.create_message')
    @patch('gmail_manager.GmailManager.authenticate_gmail')
    def test_send_email_http_error(self, mock_authenticate_gmail, mock_create_message):
        mock_authenticate_gmail.return_value = Mock()
        manager = GmailManager()
        mock_create_message.side_effect = HttpError(Mock(), b'error')
        
        with self.assertLogs(level='INFO') as log:
            manager.send_email('sender@example.com', 'recipient@example.com', 'Test Subject', 'Test Message')
            self.assertIn('An error occurred:', log.output[0])

if __name__ == '__main__':
    unittest.main()
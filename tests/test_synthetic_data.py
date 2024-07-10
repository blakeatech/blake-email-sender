import unittest
from unittest.mock import patch, Mock
import pandas as pd
import synthetic_data  # Assuming the code is saved in a file named synthetic_data.py

class TestSyntheticData(unittest.TestCase):

    @patch('synthetic_data.openai.OpenAI')
    @patch('synthetic_data.os.getenv')
    def test_generate_synthetic_example(self, mock_getenv, mock_openai):
        mock_getenv.return_value = 'fake_api_key'
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices[0].message.content = '{"email": "test@example.com", "response": "This is a test."}'
        mock_client.chat.completions.create.return_value = mock_response

        result = synthetic_data.generate_synthetic_example("Test prompt")
        
        self.assertIsNotNone(result)
        self.assertIn('email', result)
        self.assertIn('response', result)

    @patch('synthetic_data.generate_synthetic_example')
    def test_generate_synthetic_data(self, mock_generate_synthetic_example):
        mock_generate_synthetic_example.return_value = {"email": "test@example.com", "response": "This is a test."}
        
        synthetic_examples = []
        for _ in range(10):  # Reduced sample size for testing purposes
            example = synthetic_data.generate_synthetic_example("Test prompt")
            if example:
                synthetic_examples.append(example)
        
        dataframe = pd.DataFrame(synthetic_examples)
        self.assertFalse(dataframe.empty)
        self.assertIn("email", dataframe.columns)
        self.assertIn("response", dataframe.columns)

if __name__ == '__main__':
    unittest.main()
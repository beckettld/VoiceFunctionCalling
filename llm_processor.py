import requests
import json

class LLMProcessor:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint

    def process_text(self, text):
        headers = {'Content-Type': 'application/json'}
        data = {"text": text}
        try:
            response = requests.post(self.api_endpoint, json=data, headers=headers)
            print(f"Request to LLM: {self.api_endpoint} with text: {json.dumps(data)}")
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text}")

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error processing text: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

def parse_commands(text):
    return text.split('. ')

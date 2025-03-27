# src/howdoyoufindme/clean_json.py

import json
import re


def extract_json_string(response: str) -> str:
    """
    Attempts to extract just the JSON part from a string.
    Finds the first '{' and the matching last '}' and returns the substring.
    Also handles potential formatting issues.
    """
    # Find JSON structure beginning and end
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object braces found in response.")
    
    json_str = response[start_idx:end_idx + 1]
    
    # Clean up common formatting issues
    # Remove trailing commas in arrays
    json_str = re.sub(r',(\s*])', r'\1', json_str)
    # Remove trailing commas in objects
    json_str = re.sub(r',(\s*})', r'\1', json_str)
    
    try:
        # Verify it's valid JSON but return the string
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        problem_char = e.pos
        context = json_str[max(0, problem_char - 50):min(len(json_str), problem_char + 50)]
        print(f"JSON parse error near: ...{context}...")
        raise ValueError(f"Failed to parse JSON after cleaning: {str(e)}. Error location: {context}")

def clean_and_parse_json(response: str) -> dict:
    """
    Clean and parse JSON from a string response.
    Handles common formatting issues and returns a parsed JSON object.
    """
    try:
        # First try direct parsing
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            # Try to extract and clean JSON, then parse it
            cleaned_str = extract_json_string(response)
            return json.loads(cleaned_str)  # Parse the cleaned string into a dict
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Full response that failed to parse: {response}")
            raise ValueError(f"Failed to parse JSON after cleaning: {str(e)}. Content: {response[:100]}...")
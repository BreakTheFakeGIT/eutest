import json
import requests

def get_response(prompt):
    """
    To get response from the model via API call from Ollama and generate response from LLM with the given prompt.
    
    Parameters:
    prompt (str): The prompt to generate response.
    
    Returns:
    str: The generated response.
    """
    
    url = "http://127.0.0.1:11434/v1/completions"
    headers = {
        "Content-Type": "application/json"
        }
    data = {
        "prompt": prompt,
        "model": "hf.co/pswitala/pllum-8b-instruct-q5_k_m_gguf:Q5_K_M" #"llama3.2:latest"
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["text"]
    else:
        return f"Error: {response.status_code}, {response.text}"
import os
import json
import requests
import openai
from google import genai
import anthropic
import ollama
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# --- Unified LLM Client ---

def run_llm_call(prompt: str, provider: str, model: str):
    """
    A unified function to make a call to a specified LLM provider.

    Args:
        prompt (str): The user's prompt.
        provider (str): The LLM provider ('openai', 'gemini', 'anthropic', 'ollama').
        model (str): The specific model to use.

    Returns:
        A string containing the LLM's response.
    """
    print(f"--- Calling Provider: {provider.upper()}, Model: {model} ---")
    try:
        if provider == "openai":
            if not model:
                model = "gpt-4o"
            return _call_openai(prompt, model)
        elif provider == "gemini":
            if not model:
                model = "gemini-2.5-flash"
            return _call_gemini(prompt, model)
        elif provider == "anthropic":
            if not model:
                model = "claude-3-5-sonnet-20241022"
            return _call_anthropic(prompt, model)
        elif provider == "ollama":
            if not model:
                model = "gemma3:1b"
            return _call_ollama(prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        return f"Error calling {provider}: {e}"

# --- Provider-Specific Implementations ---

def _call_openai(prompt, model):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def _call_openai_rest(prompt, model):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get("OPENAI_API_KEY")}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def _call_gemini(prompt, model):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text

def _call_gemini_rest(prompt, model):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent" 
    headers = {
        "x-goog-api-key": os.environ.get("GEMINI_API_KEY"),
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# print(_call_gemini_request("what is the capital of japan?", "gemini-2.5-flash"))


def _call_anthropic(prompt, model):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def _call_anthropic_rest(prompt, model):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"  # Required API version header
    }
    
    data = {
        "model": model,
        "max_tokens": 1024, ## also required
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()["content"][0]["text"]

def _call_ollama(prompt, model):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response['message']['content']

def _call_ollama_rest(prompt, model):
    # Assumes Ollama is running on http://localhost:11434
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, # Ensure a single response object
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

if __name__ == "__main__":
    test_prompt = "What is the speed of light?"

    # OpenAI
    openai_response = run_llm_call(test_prompt, "openai", "gpt-4o")
    print(f"Response: {openai_response}\n")

    # Gemini
    gemini_response = run_llm_call(test_prompt, "gemini", "gemini-2.5-flash")
    print(f"Response: {gemini_response}\n")

    # Anthropic
    anthropic_response = run_llm_call(test_prompt, "anthropic", "claude-3-5-sonnet-20241022")
    print(f"Response: {anthropic_response}\n")

    # Ollama (make sure Ollama is running and you have the model)
    ollama_response = run_llm_call(test_prompt, "ollama", "gemma3:1b")
    print(f"Response: {ollama_response}\n")

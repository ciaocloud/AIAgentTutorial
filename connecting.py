import os
import json
import requests
import openai
import google.generativeai as genai
import anthropic
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
            return _call_openai(prompt, model)
        elif provider == "gemini":
            return _call_gemini(prompt, model)
        elif provider == "anthropic":
            return _call_anthropic(prompt, model)
        elif provider == "ollama":
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

def _call_gemini(prompt, model):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    client = genai.GenerativeModel(model_name=model)
    response = client.generate_content(prompt)
    return response.text

def _call_anthropic(prompt, model):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def _call_ollama(prompt, model):
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

# --- Demonstration ---
if __name__ == "__main__":
    test_prompt = "What is the speed of light?"

    # OpenAI
    openai_response = run_llm_call(test_prompt, "openai", "gpt-4o")
    print(f"Response: {openai_response}\n")

    # Gemini
    gemini_response = run_llm_call(test_prompt, "gemini", "gemini-1.5-flash")
    print(f"Response: {gemini_response}\n")

    # Anthropic
    anthropic_response = run_llm_call(test_prompt, "anthropic", "claude-3-sonnet-20240229")
    print(f"Response: {anthropic_response}\n")

    # Ollama (make sure Ollama is running and you have the model)
    ollama_response = run_llm_call(test_prompt, "ollama", "llama3")
    print(f"Response: {ollama_response}\n")

# Tutorial 1: Connecting to LLM APIs

Welcome to the first tutorial in our series on building an AI Agent! The foundational step for any agent is communicating with a Large Language Model (LLM).

In this guide, we will write a Python script from scratch that can connect to major LLM providers (OpenAI, Google, Anthropic) and a local model (via Ollama), all through a single, unified interface.

## Prerequisites

1.  **Python 3.8+**: Ensure you have a modern version of Python installed.
2.  **API Keys**: You'll need API keys from each cloud provider. It is recommended to save your API keys in a `.env` file as shown in the setup section.
    *   [OpenAI](https://platform.openai.com/api-keys)
    *   [Google AI Studio (Gemini)](https://aistudio.google.com/app/api-keys)
    *   [Anthropic (Claude)](https://console.anthropic.com/settings/keys)
3.  **Install `uv` (Python Package Manager)**: We will use `uv` (https://github.com/astral-sh/uv) to manage our virtual environment and dependencies. It's a modern, extremely fast replacement for `pip` and `venv`. Install it by running the following command in your terminal on mac and Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    or on windows:
    ```
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

4.  **(Local) Ollama**: For running local models, install [Ollama](https://ollama.com/) and pull a model (e.g., `ollama pull llama3`).

## Step 1: Project Setup

Let's set up our project directory and environment from scratch. Open your terminal and follow these commands.

1.  **Create the Project Directory**

    ```bash
    mkdir AiAgentTutorial
    cd AiAgentTutorial
    uv init
    ```

2.  **Initialize a Python Virtual Environment**

    We'll use `uv`, a fast, modern Python package manager. This command creates a virtual environment in a `.venv` folder.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install Required Libraries**

    Now, we'll use `uv add` to install the necessary libraries. This automatically adds them to your `pyproject.toml` and installs them into your virtual environment.

    ```bash
    uv add openai google-genai anthropic requests python-dotenv ollama
    ```

4.  **Create a `.env` file for API Keys**

    Create a file named `.env` to securely store your API keys. Our Python script will load these keys so you don't have to hard-code them.

    **Important:** Add `.env` to your `.gitignore` file to prevent accidentally committing your secrets.

    Your `.env` file should look like this:

    ```
    OPENAI_API_KEY="sk-..."
    GEMINI_API_KEY="AIza..."
    ANTHROPIC_API_KEY="sk-ant-..."
    ```

With the project structure and environment in place, we are ready to start writing the Python client.

## Step 2: Writing the Code Connecting to LLM APIs (`llm_client.py`)

Now, let's build our Python script piece by piece. 

### Connecting to OpenAI

Let's write some code in Python to connect to the OpenAI API. 

**How it works:**
1.  We instantiate the `OpenAI` client, which takes the `OPENAI_API_KEY` environment variable as the `api_key` argument.
2.  We structure the input as a list of messages. The `"role": "user"` is important for defining who is speaking.
3.  We call `client.chat.completions.create()` to get the response.
4.  The actual text content is nested inside the response object, which we access via `response.choices[0].message.content`.

```python
import os
import openai

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

system_message = "You are a helpful assistant."
user_prompt = "What is the speed of light in a vacuum?"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
model = "gpt-4o"
response = client.chat.completions.create(
    model=model,
    messages=messages,
)
print(response.choices[0].message.content)
```

### Connecting to Google Gemini

The Google Gemini library offers two primary ways to interact with the model, each suited for different use cases: `generate_content` for single, stateless requests, and `chat` for stateful, multi-turn conversations. In a high level, think of `generate_content()` as a stateless calculator. You give it an input, and it will respond with an output to you. It has no memory of your previous calculation. It's a simple, one-off transaction.
On the other hand, think of `chat()` as a stateful conversation with a person. You start a chat, and that person (the chat object) remembers everything you've both said. You only need to provide your new message, and they handle the context.

#### `model.generate_content` (For Single Turns)

This is the simplest way to interact with the model. It is stateless.
   * What it is: A direct, one-shot call to the model. You can pass a simple string directly to the function. The model has no memory of any previous calls you made.
   * When to use it:
       * Simple, single-turn tasks: "Summarize this text," "Translate this sentence," "What is the capital of France?"
       * When you want to manage the conversation history yourself manually

**How it works:**
1.  We configure the library with our `GEMINI_API_KEY`.
2.  We instantiate a client.
3.  We call `client.models.generate_content()` with a string prompt.
4.  The response object has a `.text` attribute for easy access to the result.

```python
import os
from google import genai

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model="gemini-2.0-flash"

response = client.models.generate_content(
    model=model,
    contents=user_prompt)
print(response.text)

# First turn
prompt1 = "What is the capital of Japan?"
print(f"User: {prompt1}")
response1 = client.models.generate_content(
    model=model,
    contents=prompt1)
print(f"Assistant: {response1.text}\n")

# Second turn
prompt2 = "What is the main airport there?"
print(f"User: {prompt2}")
response = client.models.generate_content(
    model=model,
    contents=prompt2)
print(f"Assistant: {response2.text}\n")
```

#### `chat.send_message` (For Conversations)

This method is designed for building chatbots. It creates a `ChatSession` object that automatically keeps track of the conversation history for you.

**How it works:**
1.  We instantiate the client as before.
2.  We call `client.chats.create()` to create a `chat` object.
3.  We use `chat.send_message()` to send our prompt. The `chat` object handles appending both the user's message and the model's response to its internal history.
4.  For subsequent calls, you just send the new message, and the model will have the full context.

```python
import os
from google import genai

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model="gemini-2.0-flash"

chat = client.chats.create(model=model, history=[])

# First turn
prompt1 = "What is the capital of Japan?"
print(f"User: {prompt1}")
response1 = chat.send_message(prompt1)
print(f"Assistant: {response1.text}\n")

# Second turn
prompt2 = "What is the main airport there?"
print(f"User: {prompt2}")
response2 = chat.send_message(prompt2)
print(f"Assistant: {response2.text}")
```

By having both functions, you can clearly see the difference between a stateless request and a stateful conversation within the same API. So which one should You use? Our suggestion is to use `model.generate_content` for simple tasks; use `chat.send_message` when you are building an interactive chatbot and want the convenience of automatic history management; and use `model.generate_content` and append to the history if you want to have full control over the context sent to the model, this developer-managed state approach is very powerful for advanced use cases, and is more consistent with most modern LLM APIs such as OpenAI's and Anthropic's.

### Connecting to Anthropic Claude

Anthropic's API design is very similar to OpenAI's, making it easy to adapt to. It uses a stateless, chat-centric endpoint that requires you to manage the conversation history.

**How it works:**
1.  We instantiate the `Anthropic` client, which reads the `ANTHROPIC_API_KEY` from your environment.
2.  We structure the input as a `messages` list, just like with OpenAI.
3.  The main API call is `client.messages.create()`.
4.  A key difference is the response format. The content is a list of blocks. For a simple text response, we access it via `response.content[0].text`.

Add the following function to your script:

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

model = "claude-3-5-sonnet-20241022"
response = client.messages.create(
    model=model,
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
)
print(response.content[0].text)
```

### Connecting to a Local Ollama Model

Finally, let's connect to a local model running via Ollama. 

The official `ollama` Python library provides a clean, synchronous client that abstracts away the HTTP requests.

**How it works:**
1.  We import the `ollama` library.
2.  We call `ollama.chat()`, which handles the connection to the local server (`http://localhost:11434`) for us.
3.  The function takes the `model` name and a `messages` list, following the same structure as the OpenAI API.
4.  We access the response content from the returned dictionary: `response['message']['content']`.


```python
import ollama

model = "gemma3"
prompt = "What is the capital of Japan?"

response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": prompt}],
)
print(response['message']['content'])
```

Alternatively, we can interact with Ollama's API directly using the `requests` library. While the `ollama` SDK is convenient, it's useful to understand that it's just making HTTP requests under the hood. This is particularly useful if you prefer not to add the `ollama` SDK dependency or need more fine-grained control over the HTTP request.

**How it works:**
1.  We use the `requests` library to send an HTTP `POST` request to `http://localhost:11434/api/chat`, which is the default address for Ollama's chat endpoint.
2.  We construct a JSON payload with the `model` name and a `messages` list.
3.  `response.raise_for_status()` is a good practice to ensure the HTTP request was successful (e.g., no 404 or 500 errors).
4.  We parse the JSON response and extract the content from the `message` key.

```python
import requests

model = "gemma3"
prompt = "What is the capital of Japan?"

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, # Ensure a single response object
    },
)
response.raise_for_status() # Raise an exception for bad status codes
print(response.json()["message"]["content"])
```



## Step 3: Wrapping in Functions and Putting Everything Together (`llm_client.py`)

Create a file named `llm_client.py`, and wrap each module we discussed above as a function.
Let's quickly recap the functions we have built in `llm_client.py`:

*   **`_call_openai(prompt, model)`**: Connects to OpenAI's `chat.completions.create` endpoint.
*   **`_call_gemini(prompt, model)`**: Connects to Google Gemini using the stateless `generate_content` method.
*   **`_call_anthropic(prompt, model)`**: Connects to Anthropic Claude, which uses a chat-centric, stateless API similar to OpenAI.
*   **`_call_ollama(prompt, model)`**: Connects to a local Ollama model by making direct HTTP requests using the `requests` library.

These functions provide the building blocks for interacting with various LLMs, each tailored to their specific API requirements.

### Demonstrating the Clients

Now that we have functions for each service, let's update our main execution block to test them all.

```python
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
```

Your script is now a complete, multi-provider LLM client! You have learned how to connect to the major cloud providers and a local model, and you understand the key differences in their API designs for both single-turn and conversational tasks.

In the next tutorials, we will build upon this foundation to explore advanced topics such as **Function Calling**, the **Model-Context-Protocol (MCP)**, **Retrieval-Augmented Generation (RAG)**, and more, as we continue to build our AI Agent.

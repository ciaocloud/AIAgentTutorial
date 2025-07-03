# Tutorial 2: Function Calling

In this tutorial, we'll explore **function calling**, a powerful feature that allows LLMs to interact with external tools and APIs. This is a crucial step in building AI agents that can take action in the real world.

## What is Function Calling?

Function calling enables you to describe functions to an LLM, and have the model intelligently decide when to use them. The model doesn't actually *call* the function, but instead returns a JSON object with the function name and arguments it believes are necessary to answer a user's query. You can then use this information to execute the function in your code.

## Why is it Important?

Function calling allows you to:

*   **Connect LLMs to the real world:** Access real-time information (e.g., weather, stock prices), interact with databases, or control smart devices.
*   **Build powerful tools:** Create agents that can send emails, schedule meetings, or perform any other action you can define in code.
*   **Overcome LLM limitations:** Bypass the model's knowledge cutoff and provide it with up-to-date information.

## Implementing Function Calling with OpenAI

Let's build a practical example using the OpenAI API. We'll create a function that can get the current weather for a given location.

### 1. Define the Tool

First, we need to define our function (or "tool") in a way the LLM can understand. This is done by providing a JSON schema that describes the function's name, description, and parameters.

```json
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
```

### 2. Make the API Call

Now, we'll make a call to the OpenAI API, providing the user's prompt and our tool definition.

```python
import openai
import os
import json

# Load API key from environment
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
```

### 3. Handle the Model's Response

The model's response will indicate if it wants to call a function. If so, it will include a `tool_calls` object in the response.

```python
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    # ... function calling logic here ...
```

The `tool_calls` object will contain the name of the function to call and the arguments to use.

### 4. Execute the Function

Now, we'll execute the function with the arguments provided by the model.

```python
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # ... (implementation of the weather function) ...
    return {"location": location, "temperature": "72", "unit": unit}

if tool_calls:
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    messages.append(response_message)  # extend conversation with assistant's reply

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            }
        )
```

### 5. Return the Result to the Model

Finally, we'll send the function's response back to the model to get a final, natural language answer.

```python
    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
    )  # get a new response from the model where it can see the function response
    print(second_response.choices[0].message.content)
```

This two-step process of calling the model, executing the function, and then calling the model again with the results allows the LLM to incorporate real-world information into its responses.

## Next Steps

In the next tutorial, we'll explore how to handle multiple function calls in a single turn, and how to build more complex, multi-step agents.

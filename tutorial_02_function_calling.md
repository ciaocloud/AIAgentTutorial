# Tutorial 2: Tool Calling

In this tutorial, we'll explore **tool calling**, a powerful feature that allows LLMs to interact with external tools and APIs. This is a crucial step in building AI agents that can take action in the real world.

## What is Tool Calling?

Tool calling is a powerful technique that allows a Large Language Model (LLM) to interact with the outside world. It enables the model to request the execution of external tools (such as functions, APIs, or databases) to gather information or perform actions that go beyond its own knowledge base.

This capability is crucial for building powerful AI agents because it allows you to:

*   **Connect LLMs to the real world:** Access real-time information (e.g., weather, stock prices), interact with databases, or control devices.
*   **Build powerful tools:** Create agents that can send emails, schedule meetings, or perform any other action you can define in code.
*   **Overcome LLM limitations:** Bypass the model's knowledge cutoff and provide it with up-to-date information.


### Function Calling vs. Tool Calling: What’s the Difference?

The two terms are often used interchangeably, but it is important to note that the industry has shifted to the broader term `"Tool Calling"` from the "old" term `"Function Calling"`.

-   **Function Calling** is now considered a *specific type* of tool calling. It refers to the model's ability of calling a specific, developer-defined backend function to be executed.
-   **Tool Calling** is the general capability of a model to use external tools. This includes function calling, but also built-in tools like a `code_interpreter` or `retrieval` systems.

Think of it this way: a "function" is a screwdriver, while a "tool" is the entire toolbox. All modern LLM APIs (OpenAI, Google, Anthropic) are now built around this broader "Tool Calling" framework.

## The Core Workflow of Tool Calling
  The process is a clear, multi-step dialogue between you (the user), your application, the LLM, and the external tools.

   1. User Initiates:
       * The user sends a prompt to your application (e.g., "What's the weather in Boston?").


   2. First API Call (Application to LLM):
       * Your application sends the user's prompt along with a list of available tools (e.g., `get_current_weather`) to the LLM.

   3. LLM Responds with a Tool Request:
       * The LLM analyzes the prompt and determines that the `get_current_weather` tool is needed.
       * It returns a **structured** request to your application, asking it to call that specific tool with the argument
         location="Boston".


   4. Application Executes the Tool:
       * Your application parses the LLM's request.
       * It calls the actual get_current_weather function (or API) with the argument "Boston".
       * The external tool processes the request and returns the result (e.g., a JSON object: {"temperature": "72"}).


   5. Second API Call (Application to LLM):
       * Your application receives the result from the tool, it then sends this result back to the LLM, letting it know what the outcome of the tool call was.


   6. LLM Generates Final Answer:
       * The LLM receives the tool call result, along with previous messages.
       * It uses the new information to compose a final, natural-language answer.
       * It returns the final answer to your application (e.g., "The weather in Boston is 72°F.").


   7. Application Responds to User:
       * Your application shows this final, human-readable answer back to the user.
<!-- The process is a two-step dialogue between you and the model:

1.  **Request**: You send a prompt to the model along with a list of available tools. The model analyzes the prompt and, if it determines one or more tools could help, it doesn’t generate a text response. Instead, it returns a structured JSON object containing the name of the tool(s) it wants to use and the arguments to pass to them.

2.  **Execution & Response**: Your code receives this request, executes the specified tool(s) with the given arguments, and then sends the result(s) back to the model in a subsequent call. The model then uses this new information to generate a final, informed, natural-language response to your original query.

This workflow allows the model to bridge the gap between its internal knowledge and real-time, external data or actions. -->

### How the LLM Decides to Call a Tool

The decision-making process of the LLMs is not random but a core part of the training. The models were fine-tuned to recognize when a user's prompt can be best answered by using one of the tools you have provided (with json format). It makes this decision by matching the semantic meaning of the user's query against the `description` fields in your tool schemas.

For example, when the model sees the prompt "What's the weather like in Boston?", it recognizes that this phrase aligns closely with the description of your `get_current_weather` tool ("Get the current weather in a given location"). This match prompts it to trigger a tool call.

This native tool-use capability is a streamlined alternative to more complex agentic frameworks like **ReAct (Reasoning and Acting)**. While ReAct requires the model to think aloud (i.e., explicitly generate "thought" and "action" steps as text), tool calling integrates this reasoning process directly into the model. The LLM internally evaluates the user prompt and matches it with tool descriptions, returns a structured tool-call request if a tool calling choice is made. Stay tuned on ReAct and tool chaining later.

### Controlling the Model's Behavior

While the default behavior is powerful, sometimes you need more control. Most APIs provide different mode choices (e.g., `tool_choice` parameter in OpenAI) that lets you guide the model's decision:

-   **`"auto"` (Default):** This is the standard mode. The model has the autonomy to decide whether to call a tool and make the choice.

-   **`none`:** This prevents the model from calling any tools entirely.

-   **`custom`:** This **forces** the model to call a *specific* tool, i.e., you can direct the model's behavior using the `tool_choice` parameter to force a specific function call (`{"type": "function", "function": {"name": "..."}}`)


## Implementing Tool Calling with OpenAI

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

We then define our tool as a function in our code, which would be loaded into the application's memory so that the application can execute it during running. Note here we provided a simplest function that simulates the actual tool call, which could be an API call to a weather service in the real-world scenario. We also store the function name and the actual function as a key-value pair in `availab`

```python
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # ... (implementation of the weather function) ...
    return {"location": location, "temperature": "85", "unit": unit, "forecast": ["sunny", "windy"]
}

available_functions = {
    "get_current_weather": get_current_weather,
}
```

### 2. Make the API Call and Handle the Model's Response

Now, we'll make a call to the OpenAI API, providing the user's prompt and our tool definition.

```python
import openai
import os
import json

# Load API key from environment
# load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-4o"

user_msg = "What's the weather like in San Jose, CA?"
messages = [{"role": "user", "content": user_msg}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
response_message = response.choices[0].message
if response_message.tool_calls:
    # ... tool calling logic here ...
```
The model's response will indicate if it wants to call a tool. If so, it will include a `tool_calls` object in the response. The `tool_calls` object will contain the name of the tool(s) to call and the arguments to use.

### 3. Execute the Function(s)

It is standard practice to loop through `response_message.tool_calls`, because modern LLMs can request multiple tool calls in a single turn (a concept known as parallel tool calling). By iterating through the list, you can execute each requested tool and append its result to your messages list, preparing a complete context for the final step.

```python
# ... (inside the if block) ...
    messages.append(response_message)  # extend conversation with the LLM's reply

    for tool_call in response_message.tool_calls: # Loop through all requested tool calls
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_result = function_to_call(**function_args) # executing get_current_weather(**function_args) 
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_result),
            }
        )
```

### 4. Return the Result to the Model

Finally, we'll send the function's result back to the model to get a final, natural language answer.

```python
    final_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
    )  # get a new response from the model where it can see the function response
    print(final_response.choices[0].message.content)
```

## Implementing Function Calling with Google Gemini

While OpenAI's tool calling requires you to be explicit and responsible for providing a JSON object that strictly defines the function's name, description, and parameters, the Google Gemini Python SDK gives you the option to simplify this by doing the work for you. 
Instead of a JSON schema, you can pass the actual Python function object directly to the model, and the library then **infers** the schema from your code. 

### 1. Define the Tool:
The Gemini SDK's ability to accept a function object is a high-level convenience wrapper. Under the hood, the SDK converts the function into a JSON schema before sending it to the Gemini API. In other words, it purely relies on the function's signature and docstring, thus we need to adopt type hints (also known as type annotations) in the function declaration and docstrings.

```python
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """
    Get the current weather in a given location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (str): The unit of temperature, either "celsius" or "fahrenheit".

    Returns:
        str: A JSON string with the weather information.
    """
    print(f"Calling weather tool for {location} in {unit}...")
    weather_info = {
        "location": location,
        "temperature": "85",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)
```
An advantage of this is that we now keep our function definition and the schema seen by the model automatically in sync. Later when you pass the `get_current_weather ` function object to the model, the SDK would extract the following:
   1. `name`: It takes the function's name: "get_current_weather".
   2. `description`: It takes the function's docstring: "Get the current weather in a given location.". This is why clear docstrings are critical when using this method.
   3. `parameters`: It inspects the function's arguments and type hints (location: str, unit: str) to build the parameter schema. It also recognizes default values.

You can bypass the automatic conversion and provide the schema explicitly if you need more control or are not using Python. This is done by constructing a `Tool` object. This approach is more verbose but is functionally equivalent to how OpenAI and others work.
```python
import google.genai as genai
from google.genai import types

# Define function schema explicitly in the FunctionDeclaration object
weather_function_declaration = types.FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location": types.Schema(
                type=types.Type.STRING,
                description="The city and state, e.g. San Francisco, CA"
            ),
            "unit": types.Schema(
                type=types.Type.STRING,
                enum=["celsius", "fahrenheit"]
            )
        },
        required=["location"]
    )
)
# Create Tool object
weather_tool = types.Tool(
    function_declarations=[weather_function_declaration]
)

## Then later you would pass this to the LLM with generate_content, assuming the client is established.
```

### 2. Make the API Call and Handle the Model's Response

```python
import os
from google import genai
from google.genai import types

# Load API key from environment
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
model="gemini-2.0-flash"

user_msg = "What's the weather like in San Jose, CA?"
response = client.models.generate_content(
    model=model,
    contents=user_msg,
    # tools=[get_current_weather] ## direct function reference
)
```
If you have created a `Tool` object explicitly (and verbosely), we can also pass it into `generate_content`:
```python
response = client.models.generate_content(
    model=model,
    contents=user_msg,
    tools=[weather_tool] ## types.Tool objects
)
```

Gemini's response will contain a `function_call` object if it determines a tool should be used.

```python
response_message = response.candidates[0].content
if response_message.parts[0].function_call:
    function_call = response_message.parts[0].function_call
    # ... function calling logic here ...
```

### 3. Execute the Function and Return the Result

The process of executing the function and returning the result to the model is similar to OpenAI's.

```python
# ... (inside the if block) ...
    if function_call.name == "get_current_weather":
        ## Execute actual function
        function_result = get_current_weather(
            location=function_call.args["location"],
            unit=function_call.args.get("unit", "fahrenheit"),
        )
        ## Continue the conversation with the result
        response = model.generate_content(
            model=model,
            content=[
                user_msg,
                response_message, # Include previous messages
                {
                    "tool_response": {
                        "name": "get_current_weather",
                        "response": json.loads(function_result),
                    }
                },
            ],
        )
        print(response.text)
```

## Implementing Function Calling with Anthropic Claude

Anthropic's approach to function calling is similar to OpenAI's, using a tool schema and a two-step process.

### 1. Define the Tool

The tool schema is slightly different from OpenAI's, but serves the same purpose.

```json
tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]
```

### 2. Make the API Call and Handle the Response

The API call and response handling are similar to OpenAI's, with minor differences in the response object.

```python
import anthropic

# client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

messages = [{"role": "user", "content": "What's the weather like in Boston?"}]

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=messages,
    tools=tools,
)

if response.stop_reason == "tool_use":
    tool_calls = [c for c in response.content if c.type == "tool_use"]
    # ... function calling logic here ...
```

### 3. Execute the Function and Return the Result

The final step is to execute the function and send the results back to the model.

```python
# ... (inside the if block) ...

    tool_outputs = []
    for tool_call in tool_calls:
        # ... (execute the function as in the Python file) ...

    messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_outputs[0]["tool_use_id"],
                    "content": str(tool_outputs[0]["content"]),
                }
            ],
        }
    )

    second_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=messages,
    )
    print(second_response.content[0].text)
```

## Implementing Function Calling with Ollama

Ollama also supports function calling with local models.

### 1. Make the API Call

```python
import ollama

# client = ollama.Client()

response = client.chat(
    model="llama3",
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
    tools=tools, # Same tool schema as OpenAI
)
```

### 2. Handle the Response and Execute the Function

The response handling is similar to OpenAI's, but the response format is a dictionary.

```python
response_message = response["message"]

if response_message.get("tool_calls"):
    tool_calls = response_message["tool_calls"]
    # ... (execute the function as in the Python file) ...
```

### 3. Return the Result to the Model

Finally, send the tool's output back to the model.

```python
# ... (inside the if block) ...

    messages.append(response_message)
    # ... (append tool output to messages) ...

    second_response = client.chat(
        model="llama3",
        messages=messages,
    )
    print(second_response["message"]["content"])
```

This two-step process of calling the model, executing the function, and then calling the model again with the results allows the LLM to incorporate real-world information into its responses.

## Key Differences in Tool Definitions

As you've seen, while the concept of tool calling is similar across providers, the way you define the tools can differ significantly. This is a crucial hurdle when building an AI agent that can work with multiple LLMs.

### Explicit vs. Implicit Schemas

The primary difference is **explicit JSON schema definition** versus **implicit schema generation from code**.

-   **OpenAI, Anthropic, and Ollama** require you to provide a detailed **JSON schema** that explicitly defines the function's name, description, and parameters. This gives you precise control but requires you to keep the schema manually synchronized with your code.

-   **Google Gemini's Python SDK** offers a more convenient approach. You can pass the Python function object directly to the model. The library then uses **introspection** to automatically generate the necessary schema from your function's signature, docstring, and type hints. While you *can* provide an explicit JSON schema to Gemini, the introspection-based method is a popular and efficient shortcut.

### The Role of an Abstraction Layer (MCP)

Dealing with these inconsistencies is where an abstraction layer, sometimes called a **Model-Context-Protocol (MCP)**, becomes invaluable. An MCP acts as a universal translator between your application and the various LLM providers.

With an MCP, you would:

1.  **Define your tool once** using the MCP's standardized format.
2.  **The MCP handles the translation**, converting your single definition into the specific format required by each provider (e.g., generating the JSON schema for OpenAI or using introspection for Gemini).

This allows you to write cleaner, provider-agnostic code and easily switch between LLMs without having to rewrite your tool definitions.

## Next Steps

In the next tutorial, we'll explore how to handle multiple function calls in a single turn, and how to build more complex, multi-step agents.

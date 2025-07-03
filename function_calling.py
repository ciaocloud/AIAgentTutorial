

import os
import json
import openai
from google import genai
import anthropic
import ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Define the External Tool ---
# In a real-world scenario, this could be an API call to a weather service
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
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# --- Main Demonstration Function ---
def main():
    """
    Demonstrates function calling with OpenAI, Gemini, Anthropic, and Ollama.
    """
    user_prompt = "What's the weather like in Boston?"
    print(f"User Prompt: {user_prompt}\n")

# --- 2. OpenAI Function Calling ---
def run_openai_tool_call(prompt: str):
    """
    Demonstrates a tool call with the OpenAI API.
    """
    print("--- OpenAI Function Calling ---")
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Step 1: Define the tool schema for the model
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
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": prompt}]

    # Step 2: Call the model with the tools
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 3: Check if the model wants to call a tool
    if tool_calls:
        print("Model wants to call a tool...")
        # Step 4: Execute the tool and get the response
        available_tools = {"get_current_weather": get_current_weather}
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  - Function: {function_name}")
            print(f"  - Arguments: {function_args}")

            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )

            # Step 5: Send the tool response back to the model
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        
        print("Sending tool response back to the model...")
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_response = second_response.choices[0].message.content
        print(f"\nFinal Model Response: {final_response}\n")
    else:
        print("Model did not request a tool call.")
        final_response = response_message.content
        print(f"\nFinal Model Response: {final_response}\n")


# --- 3. Google Gemini Function Calling ---
def run_gemini_tool_call(prompt: str):
    """
    Demonstrates a tool call with the Google Gemini API.
    """
    print("--- Google Gemini Function Calling ---")
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[get_current_weather], # Pass the function directly
    )

    # Step 1: Call the model with the prompt
    response = model.generate_content(prompt)
    response_message = response.candidates[0].content

    # Step 2: Check if the model wants to call a tool
    if response_message.parts[0].function_call:
        print("Model wants to call a tool...")
        function_call = response_message.parts[0].function_call
        function_name = function_call.name
        function_args = function_call.args

        print(f"  - Function: {function_name}")
        print(f"  - Arguments: {dict(function_args)}")

        # Step 3: Execute the tool and get the response
        if function_name == "get_current_weather":
            function_response = get_current_weather(
                location=function_args["location"],
                unit=function_args.get("unit", "fahrenheit"),
            )

            # Step 4: Send the tool response back to the model
            print("Sending tool response back to the model...")
            response = model.generate_content(
                [
                    response_message, # Include previous message
                    {
                        "tool_response": {
                            "name": "get_current_weather",
                            "response": json.loads(function_response),
                        }
                    },
                ]
            )
            final_response = response.text
            print(f"\nFinal Model Response: {final_response}\n")
    else:
        print("Model did not request a tool call.")
        final_response = response.text
        print(f"\nFinal Model Response: {final_response}\n")


# --- 4. Anthropic Claude Function Calling ---
def run_anthropic_tool_call(prompt: str):
    """
    Demonstrates a tool call with the Anthropic Claude API.
    """
    print("--- Anthropic Claude Function Calling ---")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Step 1: Define the tool schema
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

    messages = [{"role": "user", "content": prompt}]

    # Step 2: Call the model with the tools
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )

    # Step 3: Check if the model wants to call a tool
    if response.stop_reason == "tool_use":
        print("Model wants to call a tool...")
        tool_calls = [c for c in response.content if c.type == "tool_use"]
        
        # Step 4: Execute the tool and get the response
        tool_outputs = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.input

            print(f"  - Function: {tool_name}")
            print(f"  - Arguments: {tool_input}")

            if tool_name == "get_current_weather":
                tool_response = get_current_weather(
                    location=tool_input.get("location"),
                    unit=tool_input.get("unit"),
                )
                tool_outputs.append(
                    {
                        "tool_use_id": tool_call.id,
                        "content": tool_response,
                    }
                )
        
        # Step 5: Send the tool response back to the model
        print("Sending tool response back to the model...")
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
        final_response = second_response.content[0].text
        print(f"\nFinal Model Response: {final_response}\n")

    else:
        print("Model did not request a tool call.")
        final_response = response.content[0].text
        print(f"\nFinal Model Response: {final_response}\n")


# --- 5. Ollama Function Calling ---
def run_ollama_tool_call(prompt: str):
    """
    Demonstrates a tool call with a local Ollama model.
    """
    print("--- Ollama Function Calling ---")
    client = ollama.Client()

    # Step 1: Define the tool schema
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
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": prompt}]

    # Step 2: Call the model with the tools
    response = client.chat(
        model="llama3",
        messages=messages,
        tools=tools,
    )

    response_message = response["message"]

    # Step 3: Check if the model wants to call a tool
    if response_message.get("tool_calls"):
        print("Model wants to call a tool...")
        # Step 4: Execute the tool and get the response
        available_tools = {"get_current_weather": get_current_weather}
        tool_calls = response_message["tool_calls"]
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            
            print(f"  - Function: {function_name}")
            print(f"  - Arguments: {function_args}")

            function_to_call = available_tools[function_name]
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )

            # Step 5: Send the tool response back to the model
            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                    "tool_call_id": tool_call["id"],
                }
            )
        
        print("Sending tool response back to the model...")
        second_response = client.chat(
            model="llama3",
            messages=messages,
        )
        final_response = second_response["message"]["content"]
        print(f"\nFinal Model Response: {final_response}\n")
    else:
        print("Model did not request a tool call.")
        final_response = response_message["content"]
        print(f"\nFinal Model Response: {final_response}\n")


# --- Main Demonstration Function ---
def main():
    """
    Demonstrates function calling with OpenAI, Gemini, Anthropic, and Ollama.
    """
    user_prompt = "What's the weather like in Boston?"
    print(f"User Prompt: {user_prompt}\n")

    run_openai_tool_call(user_prompt)
    run_gemini_tool_call(user_prompt)
    run_anthropic_tool_call(user_prompt)
    run_ollama_tool_call(user_prompt)

if __name__ == "__main__":
    main()


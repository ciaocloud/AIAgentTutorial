# Model Context Protocol (MCP) Tutorial

## Introduction

The Model Context Protocol (MCP) is a standardized protocol that enables AI models to securely access external tools, data sources, and services. It provides a structured way for AI assistants to interact with various resources while maintaining security and control.

## Correct MCP Architecture

MCP follows a specific architecture with three main components:

1. **MCP Host**: The LLM application (like Claude Desktop, IDEs, or AI assistants) that orchestrates interactions with the LLM
2. **MCP Client**: Maintains a 1:1 connection with a single MCP server, acts as a bridge between host and server
3. **MCP Server**: Provides context, tools, and prompts to clients

### Key Architecture Principles:
- **1:1 Client-Server Relationship**: Each client connects to exactly one server
- **Host Orchestration**: The host manages multiple clients and orchestrates LLM interactions
- **Server Specialization**: Each server provides a specific set of capabilities

## MCP Host Role and Responsibilities

The MCP Host serves as the orchestrator and has several critical responsibilities:

### 1. **LLM Integration and Orchestration**
- Manages the interaction between the LLM and external services
- Translates LLM requests into MCP protocol calls
- Formats server responses for LLM consumption
- Handles context injection and prompt augmentation

### 2. **Client Management**
- Creates and manages multiple MCP clients
- Routes requests to appropriate clients based on capabilities
- Handles client lifecycle (connection, disconnection, error recovery)

### 3. **Service Discovery and Routing**
- Maintains a registry of available services and their capabilities
- Determines which client/server should handle specific requests
- Provides unified interface to the LLM regardless of underlying services

### 4. **Context and State Management**
- Maintains conversation context across multiple service interactions
- Aggregates information from multiple sources
- Manages session state and conversation flow

### 5. **Security and Access Control**
- Enforces security policies for service access
- Manages authentication and authorization
- Implements sandboxing and resource limits

## Basic MCP Implementation

Let's start by implementing a simple MCP class structure:

```python
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MCPMessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

@dataclass
class MCPMessage:
    """Base MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class MCPError(Exception):
    """MCP-specific error"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")

class MCPServer:
    """Base MCP Server implementation"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.tools: Dict[str, callable] = {}
        self.resources: Dict[str, Any] = {}
    
    def register_tool(self, name: str, handler: callable, description: str = ""):
        """Register a tool with the server"""
        self.tools[name] = {
            'handler': handler,
            'description': description
        }
    
    def register_resource(self, name: str, data: Any, description: str = ""):
        """Register a resource with the server"""
        self.resources[name] = {
            'data': data,
            'description': description
        }
    
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP requests"""
        try:
            if message.method == "tools/list":
                return MCPMessage(
                    id=message.id,
                    result={
                        "tools": [
                            {
                                "name": name,
                                "description": tool_info['description']
                            }
                            for name, tool_info in self.tools.items()
                        ]
                    }
                )
            
            elif message.method == "tools/call":
                tool_name = message.params.get('name')
                if tool_name not in self.tools:
                    raise MCPError(-32601, f"Tool '{tool_name}' not found")
                
                tool_handler = self.tools[tool_name]['handler']
                arguments = message.params.get('arguments', {})
                
                result = await tool_handler(**arguments)
                return MCPMessage(
                    id=message.id,
                    result={"content": result}
                )
            
            elif message.method == "resources/list":
                return MCPMessage(
                    id=message.id,
                    result={
                        "resources": [
                            {
                                "name": name,
                                "description": resource_info['description']
                            }
                            for name, resource_info in self.resources.items()
                        ]
                    }
                )
            
            elif message.method == "resources/read":
                resource_name = message.params.get('name')
                if resource_name not in self.resources:
                    raise MCPError(-32601, f"Resource '{resource_name}' not found")
                
                return MCPMessage(
                    id=message.id,
                    result={
                        "contents": self.resources[resource_name]['data']
                    }
                )
            
            else:
                raise MCPError(-32601, f"Method '{message.method}' not found")
                
        except MCPError as e:
            return MCPMessage(
                id=message.id,
                error={
                    "code": e.code,
                    "message": e.message,
                    "data": e.data
                }
            )
        except Exception as e:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            )

class MCPClient:
    """MCP Client - maintains 1:1 connection with a single MCP server"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.server: Optional[MCPServer] = None
        self.request_id = 0
        self.connected = False
    
    def connect_to_server(self, server: MCPServer):
        """Connect to a single MCP server (1:1 relationship)"""
        if self.server is not None:
            raise MCPError(-32001, f"Client already connected to server '{self.server.name}'")
        
        self.server = server
        self.connected = True
    
    def disconnect(self):
        """Disconnect from the server"""
        self.server = None
        self.connected = False
    
    def _get_next_id(self) -> str:
        """Generate next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Send a request to the connected MCP server"""
        if not self.connected or self.server is None:
            raise MCPError(-32002, f"Client not connected to any server")
        
        request = MCPMessage(
            id=self._get_next_id(),
            method=method,
            params=params or {}
        )
        
        response = await self.server.handle_request(request)
        
        if response.error:
            raise MCPError(
                response.error['code'],
                response.error['message'],
                response.error.get('data')
            )
        
        return response.result

class MCPHost:
    """MCP Host - LLM application that orchestrates interactions with the LLM"""
    
    def __init__(self, name: str, llm_interface=None):
        self.name = name
        self.llm_interface = llm_interface  # Interface to the LLM
        self.clients: Dict[str, MCPClient] = {}
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        self.conversation_context = []
        self.active_session = None
    
    def add_client(self, client_name: str, server: MCPServer):
        """Add a new MCP client with 1:1 server connection"""
        client = MCPClient(client_name)
        client.connect_to_server(server)
        self.clients[client_name] = client
        
        # Register service capabilities
        self.service_registry[client_name] = {
            'server_name': server.name,
            'server_version': server.version,
            'connected': True
        }
    
    def remove_client(self, client_name: str):
        """Remove a client and disconnect from its server"""
        if client_name in self.clients:
            self.clients[client_name].disconnect()
            del self.clients[client_name]
            del self.service_registry[client_name]
    
    async def discover_capabilities(self) -> Dict[str, Any]:
        """Discover all available tools and resources from connected servers"""
        capabilities = {
            'tools': {},
            'resources': {},
            'clients': {}
        }
        
        for client_name, client in self.clients.items():
            try:
                # Get tools
                tools_result = await client.send_request("tools/list")
                tools = tools_result.get('tools', [])
                
                # Get resources
                resources_result = await client.send_request("resources/list")
                resources = resources_result.get('resources', [])
                
                capabilities['clients'][client_name] = {
                    'tools': tools,
                    'resources': resources,
                    'server': self.service_registry[client_name]['server_name']
                }
                
                # Add to global registry
                for tool in tools:
                    tool_key = f"{client_name}.{tool['name']}"
                    capabilities['tools'][tool_key] = {
                        'client': client_name,
                        'description': tool['description']
                    }
                
                for resource in resources:
                    resource_key = f"{client_name}.{resource['name']}"
                    capabilities['resources'][resource_key] = {
                        'client': client_name,
                        'description': resource['description']
                    }
                    
            except Exception as e:
                capabilities['clients'][client_name] = {
                    'error': str(e),
                    'server': self.service_registry[client_name]['server_name']
                }
        
        return capabilities
    
    async def route_request(self, service_request: str, **kwargs) -> Any:
        """Route a request to the appropriate client based on service name"""
        if '.' not in service_request:
            raise MCPError(-32600, "Invalid service request format. Use 'client.method'")
        
        client_name, method = service_request.split('.', 1)
        
        if client_name not in self.clients:
            raise MCPError(-32601, f"Client '{client_name}' not found")
        
        client = self.clients[client_name]
        return await client.send_request(method, kwargs)
    
    async def call_tool(self, tool_reference: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool via the appropriate client"""
        if '.' not in tool_reference:
            raise MCPError(-32600, "Invalid tool reference format. Use 'client.tool_name'")
        
        client_name, tool_name = tool_reference.split('.', 1)
        
        if client_name not in self.clients:
            raise MCPError(-32601, f"Client '{client_name}' not found")
        
        client = self.clients[client_name]
        result = await client.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return result.get('content')
    
    async def read_resource(self, resource_reference: str) -> Any:
        """Read a resource via the appropriate client"""
        if '.' not in resource_reference:
            raise MCPError(-32600, "Invalid resource reference format. Use 'client.resource_name'")
        
        client_name, resource_name = resource_reference.split('.', 1)
        
        if client_name not in self.clients:
            raise MCPError(-32601, f"Client '{client_name}' not found")
        
        client = self.clients[client_name]
        result = await client.send_request("resources/read", {
            "name": resource_name
        })
        return result.get('contents')
    
    async def orchestrate_llm_interaction(self, user_input: str) -> str:
        """Orchestrate interaction between user, LLM, and MCP services"""
        # 1. Analyze user input to determine if MCP services are needed
        service_needs = await self.analyze_service_requirements(user_input)
        
        # 2. Gather context from MCP services if needed
        context = await self.gather_context(service_needs)
        
        # 3. Augment prompt with MCP context
        augmented_prompt = await self.augment_prompt(user_input, context)
        
        # 4. Send to LLM (simulated)
        llm_response = await self.query_llm(augmented_prompt)
        
        # 5. Process LLM response and execute any tool calls
        final_response = await self.process_llm_response(llm_response)
        
        # 6. Update conversation context
        self.conversation_context.append({
            'user_input': user_input,
            'context_used': context,
            'llm_response': llm_response,
            'final_response': final_response
        })
        
        return final_response
    
    async def analyze_service_requirements(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine what MCP services might be needed"""
        # Simple keyword-based analysis (in practice, this would be more sophisticated)
        service_needs = {
            'weather': any(word in user_input.lower() for word in ['weather', 'temperature', 'forecast']),
            'email': any(word in user_input.lower() for word in ['email', 'send', 'message']),
            'database': any(word in user_input.lower() for word in ['user', 'data', 'query', 'search'])
        }
        return service_needs
    
    async def gather_context(self, service_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant context from MCP services"""
        context = {}
        
        for service, needed in service_needs.items():
            if needed and service in self.clients:
                try:
                    # Get available tools and resources for this service
                    client = self.clients[service]
                    tools = await client.send_request("tools/list")
                    resources = await client.send_request("resources/list")
                    
                    context[service] = {
                        'available_tools': tools.get('tools', []),
                        'available_resources': resources.get('resources', [])
                    }
                except Exception as e:
                    context[service] = {'error': str(e)}
        
        return context
    
    async def augment_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Augment the user prompt with MCP context"""
        if not context:
            return user_input
        
        prompt_parts = [user_input]
        
        if context:
            prompt_parts.append("\n\nAvailable MCP Services:")
            for service, service_context in context.items():
                if 'error' not in service_context:
                    prompt_parts.append(f"\n{service.title()} Service:")
                    tools = service_context.get('available_tools', [])
                    for tool in tools:
                        prompt_parts.append(f"  - {tool['name']}: {tool['description']}")
        
        return "\n".join(prompt_parts)
    
    async def query_llm(self, prompt: str) -> str:
        """Query the LLM with the augmented prompt (simulated)"""
        # In a real implementation, this would call the actual LLM
        if self.llm_interface:
            return await self.llm_interface.query(prompt)
        
        # Simulated LLM response
        if 'weather' in prompt.lower():
            return "I'll get the weather information for you. Let me call the weather service."
        elif 'email' in prompt.lower():
            return "I'll help you send an email. Let me use the email service."
        else:
            return "I'll help you with that request."
    
    async def process_llm_response(self, llm_response: str) -> str:
        """Process LLM response and execute any tool calls"""
        # In a real implementation, this would parse tool calls from LLM response
        # For now, we'll simulate based on content
        
        if "weather service" in llm_response.lower():
            try:
                weather = await self.call_tool("weather.get_current_weather", {"city": "New York"})
                return f"{llm_response}\n\nWeather result: {weather}"
            except Exception as e:
                return f"{llm_response}\n\nError getting weather: {str(e)}"
        
        elif "email service" in llm_response.lower():
            try:
                result = await self.call_tool("email.send_email", {
                    "to": "user@example.com",
                    "subject": "Test",
                    "body": "Test email"
                })
                return f"{llm_response}\n\nEmail result: {result}"
            except Exception as e:
                return f"{llm_response}\n\nError sending email: {str(e)}"
        
        return llm_response
```

## Weather Example Implementation

Now let's implement a practical weather service using our MCP framework:

```python
import random
from datetime import datetime, timedelta

class WeatherMCPServer(MCPServer):
    """Weather service MCP server"""
    
    def __init__(self):
        super().__init__("weather-service", "1.0.0")
        
        # Sample weather data
        self.weather_data = {
            "New York": {"temp": 22, "humidity": 65, "condition": "Sunny"},
            "London": {"temp": 15, "humidity": 80, "condition": "Cloudy"},
            "Tokyo": {"temp": 28, "humidity": 70, "condition": "Rainy"},
            "Sydney": {"temp": 18, "humidity": 55, "condition": "Partly Cloudy"}
        }
        
        # Register tools
        self.register_tool(
            "get_current_weather",
            self.get_current_weather,
            "Get current weather for a city"
        )
        
        self.register_tool(
            "get_forecast",
            self.get_forecast,
            "Get weather forecast for a city"
        )
        
        # Register resources
        self.register_resource(
            "weather_stations",
            list(self.weather_data.keys()),
            "List of available weather stations"
        )
        
        self.register_resource(
            "weather_alerts",
            ["High winds expected in coastal areas", "UV index high today"],
            "Current weather alerts"
        )
    
    async def get_current_weather(self, city: str) -> Dict[str, Any]:
        """Get current weather for a city"""
        if city not in self.weather_data:
            return {"error": f"Weather data not available for {city}"}
        
        weather = self.weather_data[city].copy()
        weather['city'] = city
        weather['timestamp'] = datetime.now().isoformat()
        
        return weather
    
    async def get_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for a city"""
        if city not in self.weather_data:
            return {"error": f"Weather data not available for {city}"}
        
        base_weather = self.weather_data[city]
        forecast = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            # Simulate forecast with some variation
            temp_variation = random.randint(-5, 5)
            humidity_variation = random.randint(-10, 10)
            
            day_weather = {
                "date": date.strftime("%Y-%m-%d"),
                "temp": base_weather["temp"] + temp_variation,
                "humidity": max(0, min(100, base_weather["humidity"] + humidity_variation)),
                "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy"])
            }
            forecast.append(day_weather)
        
        return {
            "city": city,
            "forecast": forecast
        }

# Corrected MCP Architecture Example
async def corrected_mcp_example():
    print("=== Corrected MCP Architecture Demo ===\n")
    
    # Create servers (each provides specific capabilities)
    weather_server = WeatherMCPServer()
    email_server = EmailMCPServer()
    database_server = DatabaseMCPServer()
    
    # Create MCP Host (LLM application)
    llm_host = MCPHost("Claude Desktop", llm_interface=None)
    
    # Add clients with 1:1 server connections
    llm_host.add_client("weather", weather_server)
    llm_host.add_client("email", email_server)
    llm_host.add_client("database", database_server)
    
    print("1. MCP Host Service Registry:")
    for client_name, service_info in llm_host.service_registry.items():
        print(f"   {client_name}: {service_info['server_name']} v{service_info['server_version']}")
    
    print("\n2. Service Capability Discovery:")
    capabilities = await llm_host.discover_capabilities()
    
    print(f"   Connected clients: {len(capabilities['clients'])}")
    print(f"   Available tools: {len(capabilities['tools'])}")
    print(f"   Available resources: {len(capabilities['resources'])}")
    
    print("\n   Tool Registry:")
    for tool_key, tool_info in capabilities['tools'].items():
        print(f"     {tool_key}: {tool_info['description']}")
    
    print("\n3. LLM Orchestration Demo:")
    
    # Simulate user asking about weather
    user_query = "What's the weather like in Tokyo?"
    print(f"\n   User: {user_query}")
    
    llm_response = await llm_host.orchestrate_llm_interaction(user_query)
    print(f"   LLM Response: {llm_response}")
    
    # Simulate user asking to send email
    user_query = "Send an email to my team about the weather update"
    print(f"\n   User: {user_query}")
    
    llm_response = await llm_host.orchestrate_llm_interaction(user_query)
    print(f"   LLM Response: {llm_response}")
    
    print("\n4. Direct Tool Calls:")
    
    # Direct tool call via host
    weather_result = await llm_host.call_tool("weather.get_current_weather", {"city": "London"})
    print(f"   Weather tool result: {weather_result}")
    
    # Direct resource access
    weather_stations = await llm_host.read_resource("weather.weather_stations")
    print(f"   Weather stations resource: {weather_stations}")
    
    print("\n5. Conversation Context:")
    print(f"   Context entries: {len(llm_host.conversation_context)}")
    for i, entry in enumerate(llm_host.conversation_context):
        print(f"     Entry {i+1}: {entry['user_input'][:50]}...")

# Update the main function
async def main():
    await corrected_mcp_example()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
        
        self.tables[table].append(record)
        return {"status": "inserted", "id": record['id']}

# Multi-client example
async def multi_client_example():
    print("=== Multi-Client MCP Architecture Demo ===\n")
    
    # Create servers
    weather_server = WeatherMCPServer()
    email_server = EmailMCPServer()
    database_server = DatabaseMCPServer()
    
    # Create AI assistant host
    ai_assistant = MCPHost("Advanced AI Assistant")
    
    # Add different types of clients
    ai_assistant.add_client("standard", MCPClient())
    ai_assistant.add_client("authenticated", AuthenticatedMCPClient("auth_token_123"))
    ai_assistant.add_client("caching", CachingMCPClient(cache_ttl=60))
    
    # Connect servers to different clients
    ai_assistant.connect_to_server("weather", weather_server, "caching")  # Weather benefits from caching
    ai_assistant.connect_to_server("email", email_server, "authenticated")  # Email needs authentication
    ai_assistant.connect_to_server("database", database_server, "standard")  # Database uses standard client
    
    # Demonstrate multi-client capabilities
    print("1. Server Distribution Across Clients:")
    servers_by_client = await ai_assistant.list_all_servers()
    for client_name, servers in servers_by_client.items():
        print(f"   {client_name}: {servers}")
    
    print("\n2. Comprehensive Capabilities:")
    capabilities = await ai_assistant.get_comprehensive_capabilities()
    print(f"   Total clients: {len(capabilities['clients'])}")
    print(f"   Total servers: {len(capabilities['servers'])}")
    print(f"   Total tools: {capabilities['total_tools']}")
    print(f"   Total resources: {capabilities['total_resources']}")
    
    print("\n3. Using Different Services:")
    
    # Weather (through caching client)
    print("\n   Weather Service (Caching Client):")
    weather = await ai_assistant.call_tool("weather", "get_current_weather", {"city": "Tokyo"})
    print(f"     Current weather: {weather}")
    
    # Email (through authenticated client)
    print("\n   Email Service (Authenticated Client):")
    email_result = await ai_assistant.call_tool("email", "send_email", {
        "to": "user@example.com",
        "subject": "Weather Update",
        "body": f"Current weather in Tokyo: {weather}"
    })
    print(f"     Email sent: {email_result}")
    
    # Database (through standard client)
    print("\n   Database Service (Standard Client):")
    users = await ai_assistant.call_tool("database", "query", {"table": "users"})
    print(f"     Users: {users}")
    
    print("\n4. Cross-Service Workflow:")
    # Get weather for multiple cities
    cities = ["New York", "London", "Tokyo"]
    weather_reports = []
    
    for city in cities:
        weather = await ai_assistant.call_tool("weather", "get_current_weather", {"city": city})
        weather_reports.append(f"{city}: {weather['temp']}°C, {weather['condition']}")
    
    # Send summary email
    weather_summary = "\n".join(weather_reports)
    email_result = await ai_assistant.call_tool("email", "send_email", {
        "to": "admin@example.com",
        "subject": "Daily Weather Summary",
        "body": f"Weather Summary:\n{weather_summary}"
    })
    
    print(f"   Weather summary email sent: {email_result['message_id']}")
    
    print("\n5. Client-Specific Features:")
    
    # Demonstrate caching
    print("\n   Caching Client Demo:")
    start_time = datetime.now()
    await ai_assistant.call_tool("weather", "get_current_weather", {"city": "New York"})
    first_call_time = (datetime.now() - start_time).total_seconds()
    
    start_time = datetime.now()
    await ai_assistant.call_tool("weather", "get_current_weather", {"city": "New York"})
    second_call_time = (datetime.now() - start_time).total_seconds()
    
    print(f"     First call: {first_call_time:.4f}s")
    print(f"     Second call (cached): {second_call_time:.4f}s")
    print(f"     Speedup: {first_call_time / second_call_time:.2f}x")

# Update the main example to use multi-client
async def main():
    await multi_client_example()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## Key MCP Concepts Explained

### 1. **Protocol Structure**
MCP uses JSON-RPC 2.0 as its message format, providing:
- **Request/Response Pattern**: Synchronous communication
- **Notifications**: Asynchronous messages
- **Error Handling**: Standardized error codes and messages

### 2. **Tools vs Resources**
- **Tools**: Executable functions that perform actions (e.g., `get_weather`, `send_email`)
- **Resources**: Static or dynamic data that can be read (e.g., `weather_stations`, `user_preferences`)

### 3. **Security Model**
- Servers expose only explicitly registered tools and resources
- Client cannot access server internals directly
- All communication goes through the standardized protocol

### 4. **Extensibility**
- Easy to add new tools and resources
- Servers can be specialized for different domains
- Multiple servers can be connected to a single client

## Multiple MCP Clients Architecture

### Why Multiple Clients?

Having multiple MCP clients within a single host provides several key advantages:

1. **Specialization**: Different clients can be optimized for specific types of services
2. **Security**: Sensitive services can use authenticated clients while others use standard clients
3. **Performance**: Some clients can implement caching, connection pooling, or other optimizations
4. **Isolation**: Problems with one client don't affect others
5. **Scalability**: Load can be distributed across multiple clients

### Common Multi-Client Patterns

#### 1. **Service Type Segregation**
```python
# Different clients for different service types
ai_assistant.connect_to_server("weather", weather_server, "public_apis")
ai_assistant.connect_to_server("email", email_server, "authenticated")
ai_assistant.connect_to_server("database", database_server, "internal")
```

#### 2. **Performance Optimization**
```python
# Caching client for read-heavy services
ai_assistant.connect_to_server("weather", weather_server, "caching")
ai_assistant.connect_to_server("news", news_server, "caching")

# Standard client for transactional services
ai_assistant.connect_to_server("payment", payment_server, "standard")
```

#### 3. **Security Levels**
```python
# High-security client for sensitive operations
ai_assistant.connect_to_server("banking", banking_server, "high_security")

# Standard client for general services
ai_assistant.connect_to_server("weather", weather_server, "standard")
```

#### 4. **Regional Distribution**
```python
# Different clients for different geographic regions
ai_assistant.connect_to_server("weather_us", us_weather_server, "us_region")
ai_assistant.connect_to_server("weather_eu", eu_weather_server, "eu_region")
```

### Client Specialization Examples

The tutorial demonstrates several specialized client types:

- **`AuthenticatedMCPClient`**: Automatically adds authentication tokens to requests
- **`CachingMCPClient`**: Implements intelligent caching for read operations
- **`SpecializedMCPClient`**: Base class for adding middleware and custom behavior

### Advanced Multi-Client Features

#### Client Health Monitoring
```python
class HealthMonitoringMCPClient(SpecializedMCPClient):
    async def health_check(self, server_name: str) -> Dict[str, Any]:
        try:
            await self.send_request(server_name, "health/check")
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

#### Load Balancing
```python
class LoadBalancingMCPHost(MCPHost):
    def get_best_client_for_server(self, server_name: str) -> MCPClient:
        # Implement load balancing logic
        available_clients = self.get_clients_for_server(server_name)
        return min(available_clients, key=lambda c: c.current_load)
```

## Advanced Features

### Error Handling
```python
try:
    result = await ai_assistant.call_tool("weather", "invalid_tool", {})
except MCPError as e:
    print(f"MCP Error: {e.message} (Code: {e.code})")
```

### Custom Server Implementation
```python
class DatabaseMCPServer(MCPServer):
    def __init__(self, db_connection):
        super().__init__("database-service", "1.0.0")
        self.db = db_connection
        
        self.register_tool("query", self.execute_query, "Execute database query")
        self.register_resource("schema", self.get_schema(), "Database schema")
    
    async def execute_query(self, sql: str) -> List[Dict]:
        # Execute SQL query safely
        return await self.db.execute(sql)
```

## Best Practices

1. **Server Design**:
   - Keep tools focused and single-purpose
   - Validate all inputs thoroughly
   - Handle errors gracefully
   - Use descriptive names and documentation

2. **Client Usage**:
   - Always handle MCP errors
   - Cache server capabilities when possible
   - Use appropriate timeouts for requests

3. **Security**:
   - Validate and sanitize all inputs
   - Implement proper authentication if needed
   - Use least-privilege principle for tool access

4. **Performance**:
   - Implement async operations where possible
   - Consider batching related operations
   - Use connection pooling for external services

## Conclusion

The Model Context Protocol provides a standardized way for AI assistants to interact with external services while maintaining security and control. By decomposing functionality into hosts, clients, and servers, MCP enables:

- **Modularity**: Different services can be developed independently
- **Security**: Controlled access to external resources
- **Scalability**: Easy to add new capabilities
- **Interoperability**: Standardized protocol across different implementations

This weather example demonstrates how MCP can be used to create structured, secure interfaces between AI assistants and external services, providing a foundation for building more complex AI-powered applications.
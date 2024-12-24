import os
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List
from typing import _GenericAlias  # for type hinting
from typing import get_type_hints
import inspect

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: Dict[str, Dict[str, str]]
    
    def __call__(self, *args, **kwargs) -> str:
        return self.func(*args, **kwargs)

def parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Extract parameter descriptions from docstring."""
    if not docstring:
        return {}
    
    params = {}
    lines = docstring.split('\n')
    in_params = False
    current_param = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Parameters:'):
            in_params = True
        elif in_params:
            if line.startswith('-') or line.startswith('*'):
                current_param = line.lstrip('- *').split(':')[0].strip()
                params[current_param] = line.lstrip('- *').split(':')[1].strip()
            elif current_param and line:
                params[current_param] += ' ' + line.strip()
            elif not line:
                in_params = False
    
    return params

def get_type_description(type_hint: Any) -> str:
    """Get a human-readable description of a type hint."""
    if isinstance(type_hint, _GenericAlias):
        if type_hint._name == 'Literal':
            return f"one of {type_hint.__args__}"
    return type_hint.__name__

def tool(name: str = None):
    def decorator(func: Callable[..., str]) -> Tool:
        tool_name = name or func.__name__
        description = inspect.getdoc(func) or "No description available"
        
        type_hints = get_type_hints(func)
        param_docs = parse_docstring_params(description)
        sig = inspect.signature(func)
        
        params = {}
        for param_name, param in sig.parameters.items():
            params[param_name] = {
                "type": get_type_description(type_hints.get(param_name, Any)),
                "description": param_docs.get(param_name, "No description available")
            }
        
        return Tool(
            name=tool_name,
            description=description.split('\n\n')[0],
            func=func,
            parameters=params
        )
    return decorator

@tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts currency using latest exchange rates.
    
    Parameters:
        - amount: Amount to convert
        - from_currency: Source currency code (e.g., USD)
        - to_currency: Target currency code (e.g., EUR)
    """
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            
        if "rates" not in data:
            return "Error: Could not fetch exchange rates"
            
        rate = data["rates"].get(to_currency.upper())
        if not rate:
            return f"Error: No rate found for {to_currency}"
            
        converted = amount * rate
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
        
    except Exception as e:
        return f"Error converting currency: {str(e)}"

import openai

class Agent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tools: Dict[str, Tool] = {}
    
    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
    
    def get_available_tools(self) -> List[str]:
        return [f"{tool.name}: {tool.description}" for tool in self.tools.values()]
    
    def use_tool(self, tool_name: str, **kwargs: Any) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        return tool.func(**kwargs)

    def create_system_prompt(self) -> str:
        tools_json = {
            "role": "AI Assistant",
            "capabilities": [
                "Using provided tools to help users when necessary",
                "Responding directly without tools for questions that don't require tool usage",
                "Planning efficient tool usage sequences"
            ],
            "instructions": [
                "Use tools only when they are necessary for the task",
                "If a query can be answered directly, respond with a simple message instead of using tools",
                "When tools are needed, plan their usage efficiently to minimize tool calls"
            ],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        name: {
                            "type": info["type"],
                            "description": info["description"]
                        }
                        for name, info in tool.parameters.items()
                    }
                }
                for tool in self.tools.values()
            ],
            "response_format": {
                "type": "json",
                "schema": {
                    "requires_tools": {"type": "boolean", "description": "whether tools are needed for this query"},
                    "direct_response": {"type": "string", "description": "response when no tools are needed", "optional": True},
                    "thought": {"type": "string", "description": "reasoning about how to solve the task (when tools are needed)", "optional": True},
                    "plan": {"type": "array", "items": {"type": "string"}, "description": "steps to solve the task (when tools are needed)", "optional": True},
                    "tool_calls": {"type": "array", "items": {"type": "object", "properties": {"tool": {"type": "string", "description": "name of the tool"}, "args": {"type": "object", "description": "parameters for the tool"}}}, "description": "tools to call in sequence (when tools are needed)", "optional": True}
                },
                "examples": [
                    {"query": "Convert 100 USD to EUR", "response": {"requires_tools": True, "thought": "I need to use the currency conversion tool to convert USD to EUR", "plan": ["Use convert_currency tool to convert 100 USD to EUR", "Return the conversion result"], "tool_calls": [{"tool": "convert_currency", "args": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}]}},
                    {"query": "What's 500 Japanese Yen in British Pounds?", "response": {"requires_tools": True, "thought": "I need to convert JPY to GBP using the currency converter", "plan": ["Use convert_currency tool to convert 500 JPY to GBP", "Return the conversion result"], "tool_calls": [{"tool": "convert_currency", "args": {"amount": 500, "from_currency": "JPY", "to_currency": "GBP"}}]}},
                    {"query": "What currency does Japan use?", "response": {"requires_tools": False, "direct_response": "Japan uses the Japanese Yen (JPY) as its official currency. This is common knowledge that doesn't require using the currency conversion tool."}}
                ]
            }
        }
        
        return f"""You are an AI assistant that helps users by providing direct answers or using tools when necessary.
Configuration, instructions, and available tools are provided in JSON format below:

{json.dumps(tools_json, indent=2)}

Always respond with a JSON object following the response_format schema above. 
Remember to use tools only when they are actually needed for the task."""

    def plan(self, user_query: str) -> Dict:
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": user_query}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    def execute(self, user_query: str) -> str:
        try:
            plan = self.plan(user_query)
            
            if not plan.get("requires_tools", True):
                return plan["direct_response"]
            
            results = []
            for tool_call in plan["tool_calls"]:
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                result = self.use_tool(tool_name, **tool_args)
                results.append(result)
            
            return f"""Thought: {plan['thought']}
Plan: {'. '.join(plan['plan'])}
Results: {'. '.join(results)}"""
            
        except Exception as e:
            return f"Error executing plan: {str(e)}"

agent = Agent()
agent.add_tool(convert_currency)

query_list = ["I am traveling to Japan from Serbia, I have 1500 of local currency, how much of Japanese currency will I be able to get?",
                "How are you doing?"]

for query in query_list:
    print(f"\nQuery: {query}")
    result = agent.execute(query)
    print(result)

# Python Tools and Agent Framework

This project provides a framework for creating and using tools within an agent-based system. It includes a decorator for defining tools, an agent class for managing and executing tools, and an example tool for currency conversion.

## Features

-   **Tool Definition:** Easily define tools using the `@tool` decorator.
-   **Automatic Parameter Handling:** The framework automatically extracts parameter information from function signatures and docstrings.
-   **Agent Management:** The `Agent` class manages available tools and executes them based on user queries.
-   **OpenAI Integration:** Leverages OpenAI's GPT models for planning and execution.
-   **Currency Conversion Example:** Includes a built-in tool for converting currencies using the latest exchange rates.
-   **Structured Planning:** The agent generates structured plans in JSON format, outlining the steps and tool calls required to fulfill a user query.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
    (The `requirements.txt` should contain `openai`)

3. Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY="your_api_key"
    ```

## Usage

### Defining Tools

Use the `@tool` decorator to define your own tools. The decorator automatically extracts parameter information from the function signature and docstring.

```python
from typing import Literal

@tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount from one currency to another using the latest exchange rates.

    Parameters:
        - amount: The amount to convert
        - from_currency: The source currency code (e.g., USD)
        - to_currency: The target currency code (e.g., EUR)
    """
    # ... (Implementation for currency conversion)

@tool(name="search_internet")
def search_internet(query: str, region: Literal["US", "JP", "EU"] = "US") -> str:
    """Searches the internet for the given query.

    Parameters:
        - query: The search query.
        - region: The region to search in.
    """
    # ... (Implementation for internet search)
```

### Using the Agent

1. Create an instance of the `Agent` class.
2. Add tools to the agent using `add_tool()`.
3. Use the `execute()` method to process user queries.

```python
agent = Agent()
agent.add_tool(convert_currency)
# agent.add_tool(search_internet) # Example of adding another tool

query = "Convert 100 USD to EUR"
result = agent.execute(query)
print(result)
```

### Example Queries

The agent can handle various queries, including those that require tools and those that can be answered directly.

**Query requiring tool:**

```
Query: I'm traveling from Serbia to Japan. I have 1500 in local currency, how much is that in Japanese Yen?
```

**Query not requiring tool:**

```
Query: How are you doing?
```

## Code Structure

-   **`Tool` class:** Represents a tool with its name, description, function, and parameters.
-   **`parse_docstring_params()`:** Extracts parameter descriptions from docstrings.
-   **`get_type_description()`:** Gets a human-readable description of type hints.
-   **`tool()` decorator:** Defines a tool and automatically extracts parameter information.
-   **`convert_currency()`:** Example tool for currency conversion.
-   **`Agent` class:** Manages tools, generates plans, and executes user queries.
-   **`create_system_prompt()`:** Creates a system prompt for the OpenAI model.
-   **`plan()`:** Generates a plan for executing a user query.
-   **`execute()`:** Executes a plan and returns the result.

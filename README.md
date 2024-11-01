# Dynamic Reasoning Optimization Framework for Language Models

## Overview
This project provides a **Dynamic Reasoning Optimization Framework** for working with large language models (LLMs). It enables flexible selection and execution of various reasoning strategies to optimize the performance of LLMs when interacting with complex queries. The system features different modules for reasoning, query rewriting, and code generation, making it suitable for a range of use cases, from question answering to computational reasoning.

### Key Features
- **Dynamic Reasoning Selection**: The framework dynamically chooses appropriate reasoning strategies based on the complexity and nature of the input query.
- **Chain-of-Thought Reasoning**: Provides a step-by-step breakdown of complex questions to improve interpretability.
- **Query Rewriting**: Rewrites ambiguous or underspecified queries to yield better, more context-aware responses.
- **Code Reasoning**: Generates Python code to solve computational problems or answer technical queries.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Installation Steps
1. Clone the repository


2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up the project using `setup.py`:
    ```sh
    python setup.py install
    ```

## Usage
The following sections outline how to use each reasoning module provided in the framework. Example scripts are included to help you get started quickly.

### Example: Basic Usage
To get started, you can use the provided `examples/basic_usage.py` script, which shows a simple interaction using the reasoning planner.

```python
from dynamic_reasoning.src.core.planner import ReasoningPlanner
from dynamic_reasoning.src.core.model import LLMWrapper
from dynamic_reasoning.src.config.config import PlannerConfig

# Setup a language model wrapper and planner configuration
model = LLMWrapper()  # Here, substitute with the actual LLM instance.
config = PlannerConfig(max_reasoning_steps=3)

# Initialize the reasoning planner
planner = ReasoningPlanner(model, config)

# Example query
query = "How can I determine if a given number is a prime number using Python?"

# Run the planner
selected_reasoning_path = planner.select_reasoning_path(query)
print(f"Selected Reasoning Path: {selected_reasoning_path}")

# Execute reasoning
response = planner.reasoners[selected_reasoning_path[0]].reason(query)
print("Generated Response:")
print(response)
```

### Inputs and Outputs

| Reasoning Module             | Example Input                                      | Example Output                                     |
|------------------------------|----------------------------------------------------|----------------------------------------------------|
| **Chain-of-Thought**         | "How many apples do I have if I started with 5, gave 2 to my friend, and then bought 8 more?"       | Step-by-step breakdown with an answer "11 apples"   |
| **Query Rewriting**          | "Tell me about the president."                    | "Can you provide detailed information about the current president of the United States, including their policies and recent achievements?" |
| **Code Reasoning**           | "Write a Python function that calculates the factorial of a number." | Generated Python function: `def factorial(n): ...` |
| **Batch Processing (LLMWrapper)** | List of multiple prompts                          | List of corresponding model-generated responses     |

## Modules

### 1. Core Modules
- **ReasoningPlanner**: Handles dynamic reasoning strategy selection.
- **Reasoner**: Provides an interface for generating answers using the underlying LLM.
- **LLMWrapper**: Encapsulates the language model, providing tokenization, text generation, and batch processing.

### 2. Reasoning Modules
- **ChainOfThoughtReasoner**: Implements step-by-step reasoning to solve complex queries logically.
- **QueryRewriter**: Reformulates ambiguous questions for better responses.
- **CodeReasoner**: Generates Python code snippets to solve coding-related problems.

### 3. Utility Modules
- **Metrics**: Tracks and analyzes the performance of reasoning paths.
- **Logger**: Provides enhanced logging capabilities for easier debugging and tracking.
- **Data Processor**: Handles data loading, validation, and caching.

## Changes Implemented
Based on recent analysis and suggestions, several improvements were made across different modules in the project. Below are the details:

### 1. Enhanced Validation and Error Handling
- **Core Module (`planner.py`)**: Added validation checks to confirm successful initialization of reasoning modules. Any failures during initialization are now logged as errors.
- **Reasoner (`reasoner.py`)**: Imported `PlannerConfig` directly to make configuration handling explicit and consistent, improving error handling.

### 2. Improved Reasoning Logic
- **Query Rewriting (`query_rewriting.py`)**: Integrated a semantic similarity tool to improve query rewriting quality by dynamically suggesting better alternatives based on input context.
- **Code Reasoning (`code_reasoning.py`)**: Added a missing `import ast` to support parsing Python code. Enhanced fallback logic to provide user-friendly warnings when a simplified code is generated.

### 3. Updated Logger Configuration
- **Logger (`logger.py`)**: Added advanced logging features, including a new formatter to include timestamps and log levels. Configured different logging levels to provide a development and production-friendly experience.

### 4. Data Handling Improvements
- **Data Processor (`data_processor.py`)**: Added additional data validation checks to ensure unique constraints on columns where required. Improved cache management to automatically clean up expired entries.


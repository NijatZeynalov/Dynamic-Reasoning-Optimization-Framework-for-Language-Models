from typing import Optional, Dict, Tuple
import ast
from ..core.reasoner import BaseReasoner
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CodeReasoner(BaseReasoner):
    """
    Implements code-based reasoning strategy.
    """

    def __init__(self, model):
        super().__init__(model)
        self.template = """
Let's solve this using code:
Problem: {query}

Here's a Python solution with explanation:

```python
# Step-by-step solution
"""

    def reason(self, query: str) -> str:
        """
        Generate and validate code-based solution.

        Args:
            query (str): Input query

        Returns:
            str: Code solution with explanation
        """
        try:
            # Format prompt for code generation
            prompt = self._format_prompt(query, self.template)

            # Generate initial response
            response = self.model.generate(prompt)

            # Extract and validate code
            code, explanation = self._extract_code_and_explanation(response)

            if not code:
                logger.warning("No valid code found in response")
                return self._handle_code_generation_failure(query)

            # Validate and clean code
            if not self._validate_code(code):
                logger.warning("Generated code failed validation")
                code = self._attempt_code_repair(code)

            # Format final response
            return self._format_response(code, explanation)

        except Exception as e:
            self._handle_error(e, "code reasoning")

    def _extract_code_and_explanation(self, response: str) -> Tuple[str, str]:
        """Extract code and explanation from response."""
        try:
            code_block = ""
            explanation = ""
            in_code_block = False

            lines = response.split('\n')
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue

                if in_code_block:
                    code_block += line + '\n'
                else:
                    explanation += line + '\n'

            return code_block.strip(), explanation.strip()

        except Exception as e:
            logger.error(f"Error extracting code: {str(e)}")
            return "", ""

    def _validate_code(self, code: str) -> bool:
        """
        Validate generated code.

        Args:
            code (str): Generated code

        Returns:
            bool: True if code is valid
        """
        try:
            # Parse code to check for syntax errors
            ast.parse(code)

            # Check for common code smells
            code_smells = [
                "import os",
                "import sys",
                "exec(",
                "eval(",
                "__import__",
                "subprocess",
                "open(",
            ]

            return not any(smell in code for smell in code_smells)

        except SyntaxError:
            return False

    def _attempt_code_repair(self, code: str) -> str:
        """
        Attempt to repair invalid code.

        Args:
            code (str): Invalid code

        Returns:
            str: Repaired code or original if repair fails
        """
        try:
            # Common syntax fixes
            repairs = [
                (r'print [', 'print('),
                (r'raw_input\(', 'input('),
                (r'xrange\(', 'range('),
                (r'([^"])\.keys\(\)', r'\1.keys()'),
                (r'([^"])\.values\(\)', r'\1.values()'),
                (r'([^"])\.items\(\)', r'\1.items()'),
            ]

            repaired_code = code
            for pattern, replacement in repairs:
                repaired_code = repaired_code.replace(pattern, replacement)

            # Add missing colons
            lines = repaired_code.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in
                       ['def ', 'class ', 'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except ']):
                    if not line.strip().endswith(':'):
                        lines[i] = line + ':'

            repaired_code = '\n'.join(lines)

            # Validate repaired code
            if self._validate_code(repaired_code):
                return repaired_code

            return code

        except Exception as e:
            logger.error(f"Error during code repair: {str(e)}")
            return code

    def _format_response(self, code: str, explanation: str) -> str:
        """
        Format code and explanation into final response.

        Args:
            code (str): Validated code
            explanation (str): Explanation text

        Returns:
            str: Formatted response
        """
        formatted_response = []

        # Add explanation if present
        if explanation:
            formatted_response.append("Explanation:")
            formatted_response.append(explanation)
            formatted_response.append("\nImplementation:")

        # Add code block
        formatted_response.append("```python")
        formatted_response.append(code)
        formatted_response.append("```")

        # Add usage example if possible
        try:
            example = self._generate_usage_example(code)
            if example:
                formatted_response.append("\nExample usage:")
                formatted_response.append("```python")
                formatted_response.append(example)
                formatted_response.append("```")
        except Exception:
            pass

        return "\n".join(formatted_response)

    def _generate_usage_example(self, code: str) -> Optional[str]:
        """Generate example usage of the code."""
        try:
            tree = ast.parse(code)

            # Find function definitions
            function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_defs:
                return None

            # Generate example for the first function
            func = function_defs[0]
            args = []

            for arg in func.args.args:
                if hasattr(arg, 'annotation') and arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        type_name = arg.annotation.id
                        if type_name == 'int':
                            args.append('42')
                        elif type_name == 'str':
                            args.append('"example"')
                        elif type_name == 'float':
                            args.append('3.14')
                        elif type_name == 'list':
                            args.append('[1, 2, 3]')
                        elif type_name == 'dict':
                            args.append('{"key": "value"}')
                        else:
                            args.append('None')
                else:
                    args.append('42')  # Default to int

            return f"{func.name}({', '.join(args)})"

        except Exception as e:
            logger.error(f"Error generating usage example: {str(e)}")
            return None

    def _handle_code_generation_failure(self, query: str) -> str:
        logger.warning("Generating fallback code due to failure in the original strategy.")
        """Handle cases where code generation fails."""
        logger.info("Attempting simplified code generation")

        # Use a simpler template
        simple_template = """
Write a simple Python function to solve this:
{query}

```python
def solve"""

        simple_response = self.model.generate(self._format_prompt(query, simple_template))
        code, _ = self._extract_code_and_explanation(simple_response)

        if self._validate_code(code):
            return self._format_response(code, "Here's a simple solution:")
        return "Unable to generate valid code solution."
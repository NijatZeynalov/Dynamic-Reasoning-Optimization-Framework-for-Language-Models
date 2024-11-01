from typing import List, Dict
from ..core.reasoner import BaseReasoner
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChainOfThoughtReasoner(BaseReasoner):
    """
    Implements chain-of-thought reasoning strategy.
    """

    def __init__(self, model):
        super().__init__(model)
        self.template = """
Let's solve this step by step:
Question: {query}
Thought process:
1. First, let's understand what we're being asked:
   {query}

2. Let's break this down into smaller parts:

3. Now, let's solve each part:

4. Finally, let's combine our findings:

Answer:"""

    def reason(self, query: str) -> str:
        """
        Apply chain-of-thought reasoning to the query.

        Args:
            query (str): Input query

        Returns:
            str: Reasoning result with step-by-step explanation
        """
        try:
            # Format prompt with query
            prompt = self._format_prompt(query, self.template)

            # Generate initial response
            response = self.model.generate(prompt)

            # Extract and validate reasoning steps
            reasoning_steps = self._parse_reasoning_steps(response)

            if not reasoning_steps:
                logger.warning("No clear reasoning steps found in response")
                return self._handle_unclear_reasoning(response)

            # Format final response
            final_response = self._format_response(reasoning_steps)

            if not self._validate_response(final_response):
                logger.warning("Invalid reasoning response generated")
                return self._handle_invalid_response(query)

            return final_response

        except Exception as e:
            self._handle_error(e, "chain-of-thought reasoning")

    def _parse_reasoning_steps(self, response: str) -> List[Dict]:
        """Extract structured reasoning steps from response."""
        steps = []
        current_step = None

        for line in response.split("\n"):
            line = line.strip()

            if not line:
                continue

            # Check for numbered step
            if line[0].isdigit() and "." in line:
                if current_step:
                    steps.append(current_step)
                step_num = int(line[0])
                step_content = line[line.index(".") + 1:].strip()
                current_step = {
                    "number": step_num,
                    "content": step_content,
                    "details": []
                }
            elif current_step and line.startswith(("- ", "* ")):
                # Add sub-points to current step
                current_step["details"].append(line[2:])
            elif current_step:
                # Add to current step's content
                current_step["content"] += " " + line

        if current_step:
            steps.append(current_step)

        return steps

    def _format_response(self, steps: List[Dict]) -> str:
        """Format reasoning steps into clear response."""
        formatted_response = []

        for step in steps:
            formatted_response.append(f"{step['number']}. {step['content']}")
            for detail in step["details"]:
                formatted_response.append(f"   - {detail}")

        # Add final answer if present
        final_answer = self._extract_final_answer(steps)
        if final_answer:
            formatted_response.append("\nTherefore, " + final_answer)

        return "\n".join(formatted_response)

    def _extract_final_answer(self, steps: List[Dict]) -> str:
        """Extract final conclusion from reasoning steps."""
        if not steps:
            return ""

        last_step = steps[-1]
        if "conclusion" in last_step["content"].lower() or "therefore" in last_step["content"].lower():
            return last_step["content"]
        return ""

    def _handle_unclear_reasoning(self, response: str) -> str:
        """Handle cases where reasoning steps are unclear."""
        logger.info("Attempting to reformulate unclear reasoning")

        # Extract any meaningful content
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        if not lines:
            return "Unable to provide clear reasoning steps."

        # Attempt to structure the response
        formatted_response = ["Let's reason about this:"]
        formatted_response.extend([f"{i + 1}. {line}" for i, line in enumerate(lines)])

        return "\n".join(formatted_response)

    def _handle_invalid_response(self, query: str) -> str:
        """Handle invalid reasoning responses."""
        logger.info("Attempting simpler reasoning approach")

        # Use a simpler template
        simple_template = """
Question: {query}
Let me explain this simply:
1. """

        simple_response = self.model.generate(self._format_prompt(query, simple_template))
        return self._clean_response(simple_response)
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from .model import LLMWrapper
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseReasoner(ABC):
    """
    Abstract base class for reasoning strategies.
    """

    def __init__(self, config: Optional[PlannerConfig] = None, model: LLMWrapper):
        self.model = model
        self.config = config or PlannerConfig()
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def reason(self, query: str) -> str:
        """
        Apply reasoning strategy to the given query.

        Args:
            query (str): Input query

        Returns:
            str: Reasoning result
        """
        pass

    def _format_prompt(self, query: str, template: str) -> str:
        """Format query using strategy-specific template."""
        try:
            return template.format(query=query)
        except KeyError as e:
            self.logger.error(f"Error formatting prompt: {str(e)}")
            raise

    def _validate_response(self, response: str) -> bool:
        """
        Validate the reasoning response.

        Args:
            response (str): Model response

        Returns:
            bool: True if response is valid
        """
        if not response or not response.strip():
            return False

        # Check for common failure patterns
        failure_indicators = [
            "I apologize",
            "I cannot",
            "I'm unable to",
            "Error:",
            "Unknown"
        ]

        return not any(indicator in response for indicator in failure_indicators)

    def _clean_response(self, response: str) -> str:
        """Clean and format the response."""
        if not response:
            return ""

        # Remove redundant whitespace
        response = " ".join(response.split())

        # Remove common prefix patterns
        prefixes_to_remove = [
            "Here's the answer:",
            "The answer is:",
            "Result:",
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return response

    def _get_confidence_score(self, response: str) -> float:
        """
        Estimate confidence score for the response.

        Args:
            response (str): Model response

        Returns:
            float: Confidence score between 0 and 1
        """
        # Check response length
        if len(response) < 10:
            return 0.3

        # Check for uncertainty indicators
        uncertainty_indicators = [
            "maybe",
            "probably",
            "might",
            "could be",
            "I think",
            "possibly"
        ]

        confidence = 1.0
        for indicator in uncertainty_indicators:
            if indicator in response.lower():
                confidence -= 0.1

        return max(0.1, min(confidence, 1.0))

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle and log errors during reasoning."""
        error_msg = f"Error during {context}: {str(error)}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
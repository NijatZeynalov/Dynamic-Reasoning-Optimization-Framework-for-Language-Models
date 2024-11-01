from typing import Dict, List, Optional, Tuple
import time
from ..config.config import PlannerConfig
from ..reasoning_modules.chain_of_thought import ChainOfThoughtReasoner
from ..reasoning_modules.code_reasoning import CodeReasoner
from ..reasoning_modules.query_rewriting import QueryRewriter
from ..utils.logger import get_logger
from ..utils.metrics import ReasoningMetrics
from .model import LLMWrapper

logger = get_logger(__name__)


class ReasoningPlanner:
    """
    Manages the dynamic selection and execution of reasoning strategies.
    """

    def __init__(
            self,
            model: LLMWrapper,
            config: Optional[PlannerConfig] = None
    ):
        self.model = model
        self.config = config or PlannerConfig()
        self.metrics = ReasoningMetrics()
        self.reasoners = self._initialize_reasoners()
        for key, reasoner in self.reasoners.items():
            if reasoner is None:
                logger.error(f"Failed to initialize {key} reasoner.")

    def _initialize_reasoners(self) -> Dict:
        """Initialize all reasoning modules."""
        return {
            "chain_of_thought": ChainOfThoughtReasoner(self.model),
            "code_reasoning": CodeReasoner(self.model),
            "query_rewriting": QueryRewriter(self.model)
        }

    def select_reasoning_path(self, query: str) -> List[str]:
        """
        Dynamically select the optimal reasoning path based on query complexity.

        Args:
            query (str): Input query

        Returns:
            List[str]: List of selected reasoning strategies
        """
        features = self._extract_query_features(query)
        strategy_scores = self._score_strategies(features)

        selected_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.config.max_reasoning_steps]

        return [strategy for strategy, _ in selected_strategies]

    def _extract_query_features(self, query: str) -> Dict:
        """Extract relevant features from the query."""
        return {
            "length": len(query),
            "contains_numbers": any(char.isdigit() for char in query),
            "contains_code_keywords": any(keyword in query.lower()
                                          for keyword in ["code", "function", "program", "algorithm"]),
            "complexity": self._estimate_query_complexity(query),
            "requires_explanation": any(word in query.lower()
                                        for word in ["why", "how", "explain", "describe"]),
            "is_mathematical": any(symbol in query
                                   for symbol in ["+", "-", "*", "/", "=", "<", ">"]),
        }

    def _score_strategies(self, features: Dict) -> Dict[str, float]:
        """Score each reasoning strategy based on query features."""
        scores = {}

        # Score chain of thought reasoning
        cot_score = (
                features["complexity"] * 0.4 +
                (1 if features["requires_explanation"] else 0) * 0.3 +
                (len(features["length"]) > 100) * 0.3
        )
        scores["chain_of_thought"] = cot_score * self.config.strategy_weights["chain_of_thought"]

        # Score code reasoning
        code_score = (
                (1 if features["contains_code_keywords"] else 0) * 0.5 +
                (1 if features["is_mathematical"] else 0) * 0.3 +
                (1 if features["contains_numbers"] else 0) * 0.2
        )
        scores["code_reasoning"] = code_score * self.config.strategy_weights["code_reasoning"]

        # Score query rewriting
        rewrite_score = (
                (1 - features["complexity"]) * 0.4 +
                (features["length"] < 50) * 0.3 +
                (not features["requires_explanation"]) * 0.3
        )
        scores["query_rewriting"] = rewrite_score * self.config.strategy_weights["query_rewriting"]

        return scores

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity on a scale from 0 to 1."""
        complexity = min(len(query) / 200, 1.0)

        # Adjust based on various factors
        if any(word in query.lower() for word in ["why", "how", "explain"]):
            complexity += 0.2
        if query.count(",") > 2:
            complexity += 0.1
        if any(char.isdigit() for char in query):
            complexity += 0.1

        return min(complexity, 1.0)

    def process_query(self, query: str) -> Tuple[str, Dict]:
        """
        Process a query using the optimal reasoning path.

        Args:
            query (str): Input query

        Returns:
            Tuple[str, Dict]: Final response and reasoning metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")

        # Select reasoning path
        reasoning_path = self.select_reasoning_path(query)
        logger.info(f"Selected reasoning path: {reasoning_path}")

        metadata = {
            "reasoning_path": reasoning_path,
            "intermediate_results": []
        }

        current_query = query
        final_response = None

        # Apply reasoning strategies sequentially
        for strategy in reasoning_path:
            reasoner = self.reasoners[strategy]
            start_step = time.time()

            response = reasoner.reason(current_query)
            step_time = time.time() - start_step

            metadata["intermediate_results"].append({
                "strategy": strategy,
                "input": current_query,
                "output": response,
                "time_taken": step_time
            })

            if strategy == "query_rewriting":
                current_query = response
            else:
                final_response = response

        # Update metrics
        total_time = time.time() - start_time
        self.metrics.update(query, reasoning_path, final_response, total_time)

        metadata["total_time"] = total_time
        metadata["final_response"] = final_response

        return final_response, metadata
from typing import Dict, Optional, List
import re
from ..core.reasoner import BaseReasoner
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter(BaseReasoner):
    """
    Implements query rewriting strategy to improve query clarity and specificity.
    """

    def __init__(self, model):
        super().__init__(model)
        self.template = """
Rewrite the following query to be more specific and answerable:
Original query: {query}

Consider:
1. Ambiguous terms that need clarification
2. Implied context that should be explicit
3. Complex questions that can be broken down

Rewritten query:"""

    def reason(self, query: str) -> str:
        """
        Rewrite query for improved clarity and specificity.

        Args:
            query (str): Original query

        Returns:
            str: Rewritten query
        """
        try:
            # Analyze query characteristics
            query_analysis = self._analyze_query(query)

            if query_analysis["complexity"] < 0.3 and not query_analysis["needs_clarification"]:
                logger.info("Query is already clear and specific")
                return query

            # Generate rewritten query
            prompt = self._format_prompt(query, self.template)
            response = self.model.generate(prompt)

            # Extract and validate rewritten query
            rewritten_query = self._extract_rewritten_query(response)

            if not self._validate_rewritten_query(rewritten_query, query):
                logger.warning("Rewritten query validation failed")
                return self._handle_rewriting_failure(query)

            return rewritten_query

        except Exception as e:
            self._handle_error(e, "query rewriting")

    def _analyze_query(self, query: str) -> Dict:
        """Analyze query characteristics."""
        analysis = {
            "length": len(query),
            "complexity": self._calculate_complexity(query),
            "needs_clarification": False,
            "ambiguous_terms": [],
            "missing_context": False
        }

        # Check for ambiguous terms
        ambiguous_terms = ["it", "this", "that", "they", "them", "those", "these"]
        analysis["ambiguous_terms"] = [term for term in ambiguous_terms if f" {term} " in f" {query} "]

        # Check if clarification needed
        analysis["needs_clarification"] = (
                bool(analysis["ambiguous_terms"]) or
                query.count(",") > 2 or
                query.count("and") > 2 or
                query.count("or") > 1
        )

        # Check for missing context
        context_indicators = ["the", "this", "that", "these", "those"]
        words = query.lower().split()
        analysis["missing_context"] = any(
            word in context_indicators for word in words[:2]
        )

        return analysis

    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        complexity = 0.0

        # Length-based complexity
        complexity += min(len(query) / 200, 0.4)

        # Structure-based complexity
        if "?" in query:
            complexity += 0.1
        complexity += min(query.count(",") * 0.1, 0.2)
        complexity += min(query.count("and") * 0.1, 0.2)
        complexity += min(query.count("or") * 0.15, 0.3)

        # Semantic complexity
        semantic_indicators = ["why", "how", "explain", "compare", "analyze", "relationship"]
        complexity += min(
            sum(word in query.lower() for word in semantic_indicators) * 0.1,
            0.3
        )

        return min(complexity, 1.0)

    def _extract_rewritten_query(self, response: str) -> str:
        """Extract rewritten query from model response."""
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if line and not any(
                    line.lower().startswith(prefix) for prefix in
                    ["original", "consider", "rewritten", "let's", "here"]
            ):
                return line

        return response.strip()

    def _validate_rewritten_query(self, rewritten: str, original: str) -> bool:
        """
        Validate rewritten query against original.

        Args:
            rewritten (str): Rewritten query
            original (str): Original query

        Returns:
            bool: True if rewritten query is valid
        """
        if not rewritten or len(rewritten) < len(original) / 2:
            return False

        if rewritten.lower() == original.lower():
            return False

        # Check for key information preservation
        original_keywords = self._extract_keywords(original)
        rewritten_keywords = self._extract_keywords(rewritten)

        preserved_keywords = len(set(original_keywords) & set(rewritten_keywords))
        if preserved_keywords < len(original_keywords) * 0.5:
            return False

        return True

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Remove common stop words
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        }

        # Extract words and remove punctuation
        words = re.findall(r'\w+', text.lower())
        return [word for word in words if word not in stop_words]

    def _handle_rewriting_failure(self, query: str) -> str:
        """Handle cases where query rewriting fails."""
        logger.info("Attempting simplified query rewriting")

        # Use a simpler template
        simple_template = """
Make this question more specific:
{query}

Specific version:"""

        simple_response = self.model.generate(self._format_prompt(query, simple_template))
        rewritten = self._extract_rewritten_query(simple_response)

        if self._validate_rewritten_query(rewritten, query):
            return rewritten
        return query
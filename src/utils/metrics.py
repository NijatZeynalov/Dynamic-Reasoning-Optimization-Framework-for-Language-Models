from typing import Dict, List, Optional
from collections import defaultdict
import json
import time
from datetime import datetime
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningMetrics:
    """
    Track and analyze reasoning performance metrics.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.strategy_usage = defaultdict(int)
        self.query_history = []
        self.start_time = datetime.now()
        self.strategy_timings = defaultdict(list)
        self.strategy_success = defaultdict(list)

    def update(
            self,
            query: str,
            reasoning_path: List[str],
            response: str,
            execution_time: float,
            strategy_times: Optional[Dict[str, float]] = None,
            strategy_successes: Optional[Dict[str, bool]] = None
    ) -> None:
        """
        Update metrics with new query processing data.

        Args:
            query (str): Original query
            reasoning_path (List[str]): List of strategies used
            response (str): Final response
            execution_time (float): Total execution time
            strategy_times (Dict[str, float], optional): Time taken by each strategy
            strategy_successes (Dict[str, bool], optional): Success status of each strategy
        """
        try:
            # Record query details
            query_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "reasoning_path": reasoning_path,
                "execution_time": execution_time,
                "response_length": len(response) if response else 0,
                "num_strategies": len(reasoning_path),
                "strategy_times": strategy_times or {},
                "strategy_successes": strategy_successes or {}
            }

            self.query_history.append(query_data)

            # Update strategy usage and performance metrics
            for strategy in reasoning_path:
                self.strategy_usage[strategy] += 1
                if strategy_times and strategy in strategy_times:
                    self.strategy_timings[strategy].append(strategy_times[strategy])
                if strategy_successes and strategy in strategy_successes:
                    self.strategy_success[strategy].append(strategy_successes[strategy])

            # Update general metrics
            self.metrics["execution_time"].append(execution_time)
            self.metrics["num_strategies"].append(len(reasoning_path))
            self.metrics["response_length"].append(len(response) if response else 0)

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def get_summary(self) -> Dict:
        """
        Get summary of current metrics.

        Returns:
            Dict: Summary statistics
        """
        try:
            total_queries = len(self.query_history)
            if total_queries == 0:
                return {"error": "No metrics available"}

            execution_times = np.array(self.metrics["execution_time"])

            summary = {
                "general_metrics": {
                    "total_queries": total_queries,
                    "total_runtime": (datetime.now() - self.start_time).total_seconds(),
                    "average_execution_time": float(np.mean(execution_times)),
                    "execution_time_std": float(np.std(execution_times)),
                    "execution_time_percentiles": {
                        "50th": float(np.percentile(execution_times, 50)),
                        "95th": float(np.percentile(execution_times, 95)),
                        "99th": float(np.percentile(execution_times, 99))
                    },
                    "average_strategies_per_query": float(np.mean(self.metrics["num_strategies"])),
                    "average_response_length": float(np.mean(self.metrics["response_length"]))
                },
                "strategy_metrics": self._get_strategy_metrics(),
                "recent_trends": self._get_recent_trends()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating metrics summary: {str(e)}")
            return {"error": str(e)}

    def _get_strategy_metrics(self) -> Dict:
        """Calculate detailed metrics for each strategy."""
        strategy_metrics = {}

        for strategy in self.strategy_usage:
            times = self.strategy_timings[strategy]
            successes = self.strategy_success[strategy]

            metrics = {
                "usage_count": self.strategy_usage[strategy],
                "usage_percentage": (self.strategy_usage[strategy] / len(self.query_history)) * 100
            }

            if times:
                metrics.update({
                    "average_time": float(np.mean(times)),
                    "time_std": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times))
                })

            if successes:
                success_rate = sum(successes) / len(successes)
                metrics["success_rate"] = float(success_rate * 100)

            strategy_metrics[strategy] = metrics

        return strategy_metrics

    def _get_recent_trends(self, window_size: int = 50) -> Dict:
        """Calculate trends over recent queries."""
        if not self.query_history:
            return {}

        recent_queries = self.query_history[-window_size:]

        trends = {
            "recent_average_time": float(np.mean([q["execution_time"] for q in recent_queries])),
            "recent_strategy_usage": defaultdict(int),
            "recent_success_rates": defaultdict(list)
        }

        for query in recent_queries:
            for strategy in query["reasoning_path"]:
                trends["recent_strategy_usage"][strategy] += 1
                if "strategy_successes" in query and strategy in query["strategy_successes"]:
                    trends["recent_success_rates"][strategy].append(
                        query["strategy_successes"][strategy]
                    )

        # Calculate recent success rates
        trends["recent_strategy_performance"] = {
            strategy: {
                "usage": count,
                "success_rate": float(np.mean(trends["recent_success_rates"][strategy]) * 100)
                if trends["recent_success_rates"][strategy] else None
            }
            for strategy, count in trends["recent_strategy_usage"].items()
        }

        return trends

    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            filepath (str): Path to output file
        """
        try:
            metrics_data = {
                "summary": self.get_summary(),
                "query_history": self.query_history
            }

            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            logger.info(f"Metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.strategy_usage.clear()
        self.query_history.clear()
        self.start_time = datetime.now()
        self.strategy_timings.clear()
        self.strategy_success.clear()
        logger.info("Metrics reset")
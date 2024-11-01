from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    model_name: str = "llama-2-7b"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    batch_size: int = 1
    num_beams: int = 1


@dataclass
class PlannerConfig:
    """Configuration for the reasoning planner."""
    max_reasoning_steps: int = 5
    reasoning_timeout: int = 30
    strategy_weights: Dict[str, float] = None
    min_confidence_threshold: float = 0.7
    max_iterations: int = 3

    def __post_init__(self):
        if self.strategy_weights is None:
            self.strategy_weights = {
                "chain_of_thought": 1.0,
                "code_reasoning": 1.0,
                "query_rewriting": 1.0
            }


@dataclass
class LogConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: str = "dynamic_reasoning.log"
    enable_wandb: bool = False
    wandb_project: str = "dynamic-reasoning"


@dataclass
class MetricsConfig:
    """Configuration for performance metrics."""
    track_execution_time: bool = True
    track_memory_usage: bool = True
    track_reasoning_paths: bool = True
    metrics_output_file: str = "metrics.json"
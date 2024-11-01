import logging
import sys
import os
from datetime import datetime
from typing import Optional
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): Logger name
        log_level (Optional[str]): Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if not logger.handlers:
        logger.setLevel(log_level or os.getenv('LOG_LEVEL', 'INFO'))

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(
            log_dir,
            f"reasoning_framework_{datetime.now().strftime('%Y%m%d')}.log"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Set propagation
        logger.propagate = False

    return logger


class LoggerWrapper:
    """
    Wrapper class for additional logging functionality.
    """

    def __init__(self, name: str, log_level: Optional[str] = None):
        self.logger = get_logger(name, log_level)
        self._query_start_time = None
        self._current_query = None

    def start_query(self, query: str):
        """Log the start of query processing."""
        self._query_start_time = datetime.now()
        self._current_query = query
        self.logger.info(f"Starting query processing: {query}")

    def end_query(self, success: bool = True):
        """Log the end of query processing."""
        if self._query_start_time and self._current_query:
            duration = datetime.now() - self._query_start_time
            status = "successfully" if success else "with failures"
            self.logger.info(
                f"Query processed {status} in {duration.total_seconds():.2f}s: {self._current_query}"
            )
            self._query_start_time = None
            self._current_query = None

    def log_reasoning_step(self, strategy: str, input_data: str, output_data: str):
        """Log individual reasoning steps."""
        self.logger.debug(
            f"Reasoning step - Strategy: {strategy}\n"
            f"Input: {input_data}\n"
            f"Output: {output_data}"
        )

    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self.logger.error(
            f"Error during {context}: {str(error)}",
            exc_info=True
        )

    def log_performance(self, metrics: dict):
        """Log performance metrics."""
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
        self.logger.info(f"Performance metrics - {metrics_str}")

    def log_warning(self, message: str, context: str = ""):
        """Log warnings with context."""
        if context:
            message = f"[{context}] {message}"
        self.logger.warning(message)


class QueryLogger:
    """
    Specialized logger for query processing.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._current_query = None
        self._steps = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Error processing query: {self._current_query}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        self._log_summary()

    def set_query(self, query: str):
        """Set the current query being processed."""
        self._current_query = query
        self._steps = []
        self.logger.info(f"Starting query: {query}")

    def add_step(self, strategy: str, duration: float, success: bool):
        """Add a processing step."""
        self._steps.append({
            "strategy": strategy,
            "duration": duration,
            "success": success
        })

    def _log_summary(self):
        """Log processing summary."""
        if not self._steps:
            return

        total_duration = sum(step["duration"] for step in self._steps)
        success_rate = sum(1 for step in self._steps if step["success"]) / len(self._steps)

        self.logger.info(
            f"Query processing complete - "
            f"Duration: {total_duration:.2f}s, "
            f"Success rate: {success_rate:.2%}"
        )
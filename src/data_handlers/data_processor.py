from typing import Dict, List, Optional, Union, Any
import json
import pandas as pd
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles data processing and transformation for the reasoning framework.
    """

    def __init__(self):
        self.supported_formats = ["json", "csv", "jsonl", "txt"]
        self.data_cache = {}

    def load_data(
            self,
            source: Union[str, pd.DataFrame, List[Dict]],
            format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from various sources into a pandas DataFrame.

        Args:
            source: Data source (file path, DataFrame, or list of dicts)
            format: File format if source is a file path

        Returns:
            pd.DataFrame: Loaded and processed data
        """
        try:
            if isinstance(source, pd.DataFrame):
                return source

            if isinstance(source, list):
                return pd.DataFrame(source)

            if isinstance(source, str):
                if format is None:
                    format = source.split('.')[-1].lower()

                if format not in self.supported_formats:
                    raise ValueError(f"Unsupported format: {format}")

                if format == "json":
                    return pd.read_json(source)
                elif format == "csv":
                    return pd.read_csv(source)
                elif format == "jsonl":
                    return pd.read_json(source, lines=True)
                elif format == "txt":
                    return pd.DataFrame({"text": open(source).readlines()})

            raise ValueError("Invalid data source")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess and clean a query string.

        Args:
            query (str): Raw query string

        Returns:
            str: Preprocessed query
        """
        try:
            # Remove extra whitespace
            query = " ".join(query.split())

            # Remove unnecessary punctuation
            query = query.strip(".,!?")

            # Ensure query ends with question mark if it's a question
            if any(query.lower().startswith(w) for w in ["what", "when", "where", "who", "why", "how"]):
                if not query.endswith("?"):
                    query += "?"

            return query

        except Exception as e:
            logger.error(f"Error preprocessing query: {str(e)}")
            return query

    def format_response(
            self,
            response: Any,
            output_format: str = "text",
            pretty: bool = True
    ) -> str:
        """
        Format response data into specified format.

        Args:
            response: Response data to format
            output_format: Desired output format
            pretty: Whether to pretty print the output

        Returns:
            str: Formatted response
        """
        try:
            if output_format == "json":
                if pretty:
                    return json.dumps(response, indent=2)
                return json.dumps(response)

            if output_format == "text":
                if isinstance(response, (dict, list)):
                    return json.dumps(response, indent=2 if pretty else None)
                return str(response)

            if output_format == "csv":
                if isinstance(response, pd.DataFrame):
                    return response.to_csv(index=False)
                if isinstance(response, (dict, list)):
                    return pd.DataFrame(response).to_csv(index=False)

            raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return str(response)

    def batch_process(
            self,
            queries: List[str],
            batch_size: int = 32
    ) -> List[str]:
        """
        Preprocess a batch of queries.

        Args:
            queries: List of query strings
            batch_size: Size of processing batches

        Returns:
            List[str]: Preprocessed queries
        """
        try:
            processed_queries = []

            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                processed_batch = [self.preprocess_query(q) for q in batch]
                processed_queries.extend(processed_batch)

            return processed_queries

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return queries

    def validate_data(
            self,
            data: pd.DataFrame,
            required_columns: Optional[List[str]] = None,
            data_types: Optional[Dict[str, type]] = None,
            unique_columns: Optional[List[str]] = None
            self,
            data: pd.DataFrame,
            required_columns: Optional[List[str]] = None,
            data_types: Optional[Dict[str, type]] = None
    ) -> bool:
        """
        Validate DataFrame structure and content.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            data_types: Dictionary of column name to expected data type

        Returns:
            bool: True if validation passes
        """
        try:
            if required_columns:
                missing_cols = set(required_columns) - set(data.columns)
                if missing_cols:
                    logger.error(f"Missing required columns: {missing_cols}")
                    return False

            if data_types:
                for col, dtype in data_types.items():
                    if col in data.columns:
                        if not all(isinstance(x, dtype) for x in data[col].dropna()):
                            logger.error(f"Invalid data type in column {col}")
                            return False

            if unique_columns:
                for col in unique_columns:
                    if data[col].duplicated().any():
                        logger.error(f"Duplicate values found in column {col}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False

    def cache_data(self, key: str, data: Any) -> None:
        """Cache data for reuse."""
        self.data_cache[key] = {
            'data': data,
            'timestamp': pd.Timestamp.now()
        }

    def get_cached_data(self, key: str, max_age_minutes: int = 60) -> Optional[Any]:
        """Retrieve cached data if not expired."""
        if key in self.data_cache:
            cache_entry = self.data_cache[key]
            age = (pd.Timestamp.now() - cache_entry['timestamp']).total_seconds() / 60

            if age <= max_age_minutes:
                return cache_entry['data']

            del self.data_cache[key]
        return None
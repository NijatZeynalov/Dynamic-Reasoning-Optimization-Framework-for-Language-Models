from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config.config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LLMWrapper:
    """
    Wrapper class for Language Model operations.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the input prompt.

        Args:
            prompt (str): Input text prompt
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generation_config = {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_beams": self.config.num_beams,
                "do_sample": True,
                **kwargs
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Get embeddings for input text.

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Text embeddings
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_input_embeddings()(inputs["input_ids"])
            return outputs.mean(dim=1)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts in batch.

        Args:
            prompts (List[str]): List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated texts
        """
        try:
            batch_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            generation_config = {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_beams": self.config.num_beams,
                "do_sample": True,
                **kwargs
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **batch_inputs,
                    **generation_config
                )

            return [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            raise

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Make the class callable for easy generation.
        """
        return self.generate(prompt, **kwargs)
from abc import ABC, abstractmethod
from core.logger import llm_logger


class BaseModel(ABC):
    def __init__(self, model_id, logger=llm_logger):
        self.model_id = model_id
        self.logger = logger
        self.model, self.tokenizer = self.init_model()

    @abstractmethod
    def init_model(self, **kwargs) -> tuple:
        """
        Initialize the model and tokenizer
        """

    @abstractmethod
    def run_inference(self, query: str, **kwargs) -> str:
        """
        Run inference on the model
        """

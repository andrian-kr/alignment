from llm.models.base_model import BaseModel
from langfuse import Langfuse


class BaseEvaluator:
    def __init__(
        self,
        dataset_name: str,
        model: BaseModel,
        english_eval: bool = False,
    ):
        self.english_eval = english_eval
        self.dataset_name = dataset_name
        self.model = model

        self.langfuse_client = Langfuse()
        self.dataset = self.langfuse_client.get_dataset(self.dataset_name)

    def run_evaluation(self, **kwargs):
        """
        Run evaluation on the dataset
        """

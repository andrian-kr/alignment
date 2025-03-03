import torch
from transformers import pipeline
from utils.common import get_device

from llm.models.base_model import BaseModel


class Gemma(BaseModel):
    def __init__(self):
        super().__init__("google/gemma-2-9b-it")
        self.pipeline = self.init_pipeline()

    def init_pipeline(self):
        self.logger.info(f"Initializing model {self.model_id}")
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device=get_device(),
        )
        return pipe

    def init_model(self) -> tuple:
        return None, None

    def run_inference(self, query: str, max_new_tokens: int = 512) -> str:
        torch.cuda.empty_cache()
        messages = [{"role": "user", "content": query}]

        with torch.no_grad():
            outputs = self.pipeline(
                messages,
                max_new_tokens=max_new_tokens,
            )
        gen_text = outputs[0]["generated_text"][-1]["content"]
        return gen_text.strip()

import torch
from llm.models.base_model import BaseModel
from transformers import pipeline


class Llama(BaseModel):
    def __init__(self):
        super().__init__("meta-llama/Llama-3.2-3B-Instruct")
        self.pipeline = self.init_pipeline()

    def init_pipeline(self):
        self.logger.info(f"Initializing model {self.model_id}")
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.logger.info(f"Model {self.model_id} initialized")
        return pipe

    def init_model(self) -> tuple:
        return None, None

    def run_inference(self, query: str, max_new_tokens: int = 756) -> str:
        torch.cuda.empty_cache()
        messages = [
            {"role": "user", "content": query},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
        )
        gen_text = outputs[0]["generated_text"][-1]["content"]
        return gen_text.strip()

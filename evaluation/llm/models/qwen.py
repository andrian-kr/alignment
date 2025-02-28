from llm.models.base_model import BaseModel
from vllm import LLM, SamplingParams


class Qwen(BaseModel):
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-7B-Instruct")

    def init_model(self, max_token_length: int = 10000) -> tuple:
        self.logger.info(f"Initializing model {self.model_id}")
        model = LLM(model=self.model_id, max_model_len=max_token_length)
        self.logger.info(f"Model {self.model_id} initialized")
        return model, None

    def run_inference(self, query: str, max_new_tokens: int = 512) -> str:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
        outputs = self.model.generate([query], sampling_params)
        return outputs[0].outputs[0].text.strip()

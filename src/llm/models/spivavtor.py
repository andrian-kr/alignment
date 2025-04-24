import torch
from llm.models.base_model import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.common import get_device


class Spivavtor(BaseModel):
    def __init__(self, model_id: str = "grammarly/spivavtor-xxl", **kwargs):
        super().__init__(model_id)

    def init_model(self, eval_mode: bool = True) -> tuple:
        self.logger.info(f"Initializing model {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if eval_mode:
            model.eval()

        self.logger.info(f"Model {self.model_id} initialized")
        return model, tokenizer

    def run_inference(self, query: str, max_new_tokens: int = 512, device: str | None = None) -> str:
        torch.cuda.empty_cache()
        device = device or get_device()

        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids.to(device),
                max_new_tokens=max_new_tokens,
            )
        gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        self.logger.debug(f"Generated text: {gen_text}")
        return gen_text.strip()

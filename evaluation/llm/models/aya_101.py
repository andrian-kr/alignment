"""Implementation of Aya-101 model for alignment evaluation."""

import torch
from core.exceptions import ModelInitializationError, ModelInferenceError
from llm.models.base_model import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.common import get_device


class Aya101(BaseModel):
    def __init__(self, model_id: str = "CohereForAI/aya-101", **kwargs):
        super().__init__(model_id)

    def init_model(self, eval_mode: bool = True) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if eval_mode:
            model.eval()

        return model, tokenizer

    def run_inference(self, query: str, max_new_tokens: int = 512, device: str | None = None) -> str:
        torch.cuda.empty_cache()
        device = device or get_device()

        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        self.logger.debug(f"Generated text: {gen_text}")
        return gen_text.strip()

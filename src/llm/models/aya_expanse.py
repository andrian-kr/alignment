import torch
from llm.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.common import get_device


class AyaExpanse(BaseModel):
    def __init__(self):
        super().__init__("CohereForAI/aya-expanse-8b")

    def init_model(self, eval_mode: bool = True) -> tuple:
        self.logger.info(f"Initializing model {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
        )

        if eval_mode:
            model.eval()

        self.logger.info(f"Model {self.model_id} initialized")
        return model, tokenizer

    def run_inference(self, query: str, max_new_tokens: int = 512, device: str | None = None) -> str:
        torch.cuda.empty_cache()
        device = device or get_device()

        messages = [{"role": "user", "content": query}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_padded_len = len(input_ids[0])
        gen_tokens = gen_tokens[0][prompt_padded_len:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return gen_text.strip()

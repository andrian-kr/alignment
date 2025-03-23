import gc

import torch
from core.logger import general_logger
from dotenv import load_dotenv
from evaluators.sc_101_evaluator import SocialChem101Evaluator
from evaluators.ethics_evaluator import EthicsCommonsenseEvaluator
from llm.models.aya_expanse import AyaExpanse
from llm.models.gemma import Gemma
from llm.models.llama import Llama
from llm.models.qwen import Qwen
from utils.common import login_to_hf

load_dotenv()

login_to_hf()

model_classes = [AyaExpanse, Llama, Gemma, Qwen]
for model_class in model_classes:
    model = model_class()
    general_logger.info(f"Running evaluation for model: {model.model_id}")
    evaluator = EthicsCommonsenseEvaluator(dataset_name="ethics_commonsense_deepl", model=model)
    evaluator.run_evaluation()

    del evaluator
    del model

    torch.cuda.empty_cache()
    gc.collect()

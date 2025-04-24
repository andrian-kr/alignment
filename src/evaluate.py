import gc

import torch
from core.logger import general_logger
from dotenv import load_dotenv
from evaluators.ethics_evaluator import EthicsCommonsenseEvaluator
from evaluators.mixed_pva_evaluator import MixedPVAEvaluator
from evaluators.sc_101_evaluator import SocialChem101Evaluator
from llm.models.aya_101 import Aya101
from llm.models.aya_expanse import AyaExpanse
from llm.models.gemma import Gemma
from llm.models.llama import Llama
from llm.models.qwen import Qwen
from utils.common import login_to_hf

load_dotenv()

login_to_hf()

model_classes = [Gemma, Llama, AyaExpanse, Qwen, Aya101]
evaluators = [EthicsCommonsenseEvaluator]
dataset_map = {
    EthicsCommonsenseEvaluator: "ethics_commonsense_claude",
    SocialChem101Evaluator: "sc_101_care_harm_claude",
    MixedPVAEvaluator: "mixed_bad",
}
ENGLISH_EVAL = True

for model_class in model_classes:
    model = model_class()

    for evaluator in evaluators:
        general_logger.info(f"Running evaluation {evaluator} for model: {model.model_id}")
        evaluator_instance = evaluator(
            model=model,
            dataset_name=dataset_map[evaluator],
        )
        evaluator_instance.run_evaluation()

        del evaluator_instance

        if ENGLISH_EVAL:
            evaluator_instance = evaluator(model=model, dataset_name=dataset_map[evaluator], english_eval=True)
            evaluator_instance.run_evaluation()

        del evaluator_instance

    del model

    torch.cuda.empty_cache()
    gc.collect()

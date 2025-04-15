import gc

import torch
from core.logger import general_logger
from dotenv import load_dotenv
from evaluators.ethics_evaluator import EthicsCommonsenseEvaluator
from evaluators.sc_101_pva_evaluator import SocialChem101PVAEvaluator
from evaluators.sc_101_evaluator import SocialChem101Evaluator
from llm.models.aya_101 import Aya101
from llm.models.aya_expanse import AyaExpanse
from llm.models.gemma import Gemma
from llm.models.llama import Llama
from llm.models.qwen import Qwen
from translation.spivavtor_pipeline import SpivavtorPipeline
from utils.common import login_to_hf

load_dotenv()

login_to_hf()

model_classes = [Aya101]
for model_class in model_classes:
    model = model_class()
    general_logger.info(f"Running evaluation for model: {model.model_id}")
    evaluator = SocialChem101PVAEvaluator(model=model)
    evaluator.run_evaluation()

    del evaluator
    del model

    torch.cuda.empty_cache()
    gc.collect()

# spivavtor_pipeline = SpivavtorPipeline()
# spivavtor_pipeline.improve_coherence(
#     output_file_path="evaluation/datasets/social-chem-101/social-chem-101_care-spivavtor_xxl_coherence.csv"
# )
# spivavtor_pipeline.improve_gec(
#     file_path="evaluation/datasets/ethics/ethics_commonsense_claude.csv",
#     output_file_path="evaluation/datasets/ethics/ethics_commonsense_claude2.csv",
#     input_column_name="input_ukr",
#     output_column_name="input_ukr_spivavtor",
# )

# spivavtor_pipeline.improve_gec(
#     file_path="evaluation/datasets/social-chem-101/social-chem-101_care-harm_rot-agree_4_claude.csv",
#     output_file_path="evaluation/datasets/social-chem-101/social-chem-101_care-harm_rot-agree_4_claude2.csv",
#     input_column_name="action_ukr",
#     output_column_name="action_ukr_spivavtor",
# )

from dotenv import load_dotenv
from evaluators.ethics_evaluator import EthicsCommonsenseEvaluator
from llm.models.gemma import Gemma
from utils.common import login_to_hf

load_dotenv()

login_to_hf()

gemma_model = Gemma()
ethics_evaluator = EthicsCommonsenseEvaluator(model=gemma_model)
ethics_evaluator.run_evaluation()

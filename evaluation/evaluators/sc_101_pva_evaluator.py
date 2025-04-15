import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import (
    PydanticOutputParser,
)
from langchain_openai import ChatOpenAI
from llm.models.base_model import BaseModel
from llm.prompts import (
    alignment_eval_prompt,
    sc_101_eval_prompt_pva_bad,
    sc_101_eval_prompt_pva_good,
)
from pydantic import BaseModel as PydanticBaseModel
from tqdm import tqdm
from utils.common import get_device

from evaluators.base_evaluator import BaseEvaluator

load_dotenv()

LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=256,
)


class ModelAlignment(PydanticBaseModel):
    score: int
    explanation: str


class SocialChem101PVAEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, dataset_name: str = "sc_101_care_harm_pva"):
        super().__init__(dataset_name=dataset_name, model=model)
        self.output_parser = PydanticOutputParser(pydantic_object=ModelAlignment)
        self.eval_chain = self.get_chain()
        self.results_dir = os.path.join(os.getenv("RESULTS_DIR"), dataset_name)

    def run_evaluation(self, english_eval: bool = False):
        run_name = f"{self.model.model_name}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        if english_eval:
            run_name = f"en_{run_name}"

        results = []

        for item in tqdm(self.dataset.items[:20], desc="Evaluation progress"):
            with item.observe(run_name=run_name) as trace_id:
                query = item.input["input_en"] if english_eval else item.input["input"]
                expected_output = item.expected_output["label"]

                prompt = (
                    sc_101_eval_prompt_pva_bad.format(query=query)
                    if expected_output == 0
                    else sc_101_eval_prompt_pva_good.format(query=query)
                )
                prediction = self.model.run_inference(query=prompt, device=get_device())

                self.langfuse_client.trace(
                    id=trace_id,
                    input={
                        "query": query,
                        "input_en": item.input["input_en"] if not english_eval else None,
                        "prompt": prompt,
                        "label": expected_output,
                    },
                    output={"prediction": prediction},
                )

                score, comment = self.log_score(prompt, prediction, trace_id)
                results.append(
                    {
                        "query": query,
                        "prompt": prompt,
                        "label": expected_output,
                        "prediction": prediction,
                        "llm_eval_score": score,
                        "llm_eval_explanation": comment,
                    }
                )

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{run_name}.csv", index=False)

    def get_chain(self):
        chain = alignment_eval_prompt | LLM | self.output_parser
        return chain

    def log_score(self, initial_prompt: str, prediction: str, trace_id: str):
        if prediction is None:
            self.langfuse_client.score(trace_id=trace_id, name="accuracy", value=False)
            return 0, ""
        if isinstance(prediction, str):
            response = self.eval_chain.invoke({"query": initial_prompt, "response": prediction})
            self.langfuse_client.score(
                trace_id=trace_id, name="llm_eval", value=response.score, comment=response.explanation
            )
            return response

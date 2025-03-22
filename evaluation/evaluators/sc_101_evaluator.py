from datetime import datetime

from llm.models.base_model import BaseModel
from llm.prompts import sc_101_eval_prompt, sc_101_eval_prompt_en
from tqdm import tqdm

from evaluators.base_evaluator import BaseEvaluator


class SocialChem101Evaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, dataset_name: str = "sc_101_care_harm"):
        super().__init__(dataset_name=dataset_name, model=model)

    def run_evaluation(self, english_eval: bool = False):
        run_name = f"{self.model.model_name}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        if english_eval:
            run_name = f"en_{run_name}"

        for item in tqdm(self.dataset.items, desc="Evaluation progress"):
            with item.observe(run_name=run_name) as trace_id:
                query = item.input["input_en"] if english_eval else item.input["input"]
                expected_output = item.expected_output["label"]

                prompt = (
                    sc_101_eval_prompt_en.format(query=query)
                    if english_eval
                    else sc_101_eval_prompt.format(query=query)
                )
                # TODO: add batch inference
                prediction = self.model.run_inference(query=prompt)
                self.langfuse_client.trace(
                    id=trace_id,
                    input=(
                        {
                            "query": query,
                            "expected_output": expected_output,
                        }
                        if english_eval
                        else {
                            "query": query,
                            "input_en": item.input["input_en"],
                            "expected_output": expected_output,
                        }
                    ),
                    output={"prediction": prediction},
                )
                try:
                    predicted_label = int(prediction[0])
                except ValueError:
                    predicted_label = None

                self.log_score(predicted_label, int(expected_output), trace_id)

    def log_score(self, prediction: int | None, expected_output: int, trace_id: str):
        self.langfuse_client.score(trace_id=trace_id, name="accuray", value=prediction == expected_output)

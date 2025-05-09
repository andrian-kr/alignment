from datetime import datetime

from evaluate import load
from evaluators.base_evaluator import BaseEvaluator
from langdetect import detect, detect_langs
from llm.models.base_model import BaseModel
from llm.prompts import aya_suite_eval_prompt
from tqdm import tqdm


class AyaSuiteEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, bert_score_model_type: str = "xlm-roberta-large"):
        super().__init__(dataset_name="aya_eval_ukr", model=model)
        self.bert_score = load("bertscore")
        self.bert_score_model_type = bert_score_model_type

    def run_evaluation(self):
        run_name = (
            f"{self.model.model_name}_{self.bert_score_model_type}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        for item in tqdm(self.dataset.items, desc="Evaluation progress"):
            with item.observe(run_name=run_name) as trace_id:
                query = item.input["query"]
                expected_output = item.expected_output["targets"]

                prompt = aya_suite_eval_prompt.format(query=query)
                prediction = self.model.run_inference(query=prompt)

                try:
                    languages = detect_langs(prediction)
                except Exception as e:
                    languages = str(e)

                self.langfuse_client.trace(
                    id=trace_id,
                    input={
                        "query": query,
                        "expected_output": expected_output,
                    },
                    output={"prediction": prediction, "languages": languages},
                )

                self.log_score(prediction, expected_output, trace_id)

    def log_score(self, prediction: str, expected_output: str, trace_id: str):
        bert_scores = self.bert_score.compute(
            predictions=[prediction], references=[expected_output], lang="uk", model_type=self.bert_score_model_type
        )
        try:
            main_language = detect(prediction)
        except Exception as e:
            main_language = str(e)

        self.langfuse_client.score(trace_id=trace_id, name="bert_percision", value=bert_scores["precision"][0])
        self.langfuse_client.score(trace_id=trace_id, name="bert_recall", value=bert_scores["recall"][0])
        self.langfuse_client.score(trace_id=trace_id, name="bert_f1", value=bert_scores["f1"][0])
        self.langfuse_client.score(trace_id=trace_id, name="is_ukrainian", value=main_language == "uk")

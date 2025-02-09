import os
from datetime import datetime

import torch
from evaluate import load
from huggingface_hub import login
from langdetect import detect, detect_langs
from langfuse import Langfuse
from langfuse.client import DatasetItemClient
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

print("started")

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

print("logged in to Huggingface")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATASET_NAME = "aya_eval_ukr"
print("loading")
BERT_SCORE = load("bertscore")


LANGFUSE_CLIENT = Langfuse()

MODEL_ID = "CohereForAI/aya-expanse-8b"
tokenizer_aya = AutoTokenizer.from_pretrained(MODEL_ID)
model_aya = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
)


def replace_special_tokens_with_whitespace(text: str, tokenizer: AutoTokenizer = tokenizer_aya) -> str:
    special_tokens = tokenizer.special_tokens_map.values()
    for token in special_tokens:
        text = text.replace(token, " ")
    return text.strip()


def predict(example: DatasetItemClient, device: str = DEVICE):
    torch.cuda.empty_cache()
    input_str = example.input["query"]
    formatted = (
        "Дай відповідь на питання, враховуючи контекст, якщо він присутній.\nВідповідай лише українською.\n\n"
        + input_str
    )
    messages = [{"role": "user", "content": formatted}]
    input_ids = tokenizer_aya.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    with torch.no_grad():
        gen_tokens = model_aya.generate(
            input_ids.to(device),
            max_new_tokens=512,
            do_sample=False,
        )
    prompt_padded_len = len(input_ids[0])
    gen_tokens = gen_tokens[0][prompt_padded_len:]
    gen_text = tokenizer_aya.decode(gen_tokens, skip_special_tokents=True)
    return replace_special_tokens_with_whitespace(gen_text)


def log_score(prediction: str, expected_output: str, trace_id: str):
    bert_scores = BERT_SCORE.compute(predictions=[prediction], references=[expected_output], lang="uk", device="cpu")
    main_language = detect(prediction)
    LANGFUSE_CLIENT.score(trace_id=trace_id, name="bert_percision", value=bert_scores["precision"][0])
    LANGFUSE_CLIENT.score(trace_id=trace_id, name="bert_recall", value=bert_scores["recall"][0])
    LANGFUSE_CLIENT.score(trace_id=trace_id, name="bert_f1", value=bert_scores["f1"][0])
    LANGFUSE_CLIENT.score(trace_id=trace_id, name="is_ukrainian", value=main_language == "uk")


def run_evaluation(
    dataset_name: str = DATASET_NAME,
):
    run_name = "aya-expanse-8b" + dataset_name + f"-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    dataset = LANGFUSE_CLIENT.get_dataset(dataset_name)
    for item in tqdm(dataset.items, desc="Evaluation progress"):
        with item.observe(run_name=run_name) as trace_id:
            query = item.input["query"]
            expected_output = item.expected_output["targets"]

            prediction = predict(example=item)
            languages = detect_langs(prediction)
            LANGFUSE_CLIENT.trace(
                id=trace_id,
                input={
                    "query": query,
                    "expected_output": expected_output,
                },
                output={"prediction": prediction, "languages": languages},
            )

            log_score(prediction, expected_output, trace_id)


run_evaluation()

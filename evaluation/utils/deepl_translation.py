import os

import deepl
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()
tqdm.pandas()

TRANSLATOR = deepl.Translator(auth_key=os.getenv("DEEPL_API_KEY"))


def traslate_csv(
    file_path: str,
    source_column: str,
    target_column: str,
    ouput_file_path: str,
):
    df = pd.read_csv(file_path)
    df[target_column] = df[source_column].progress_apply(lambda x: TRANSLATOR.translate_text(x, target_lang="UK").text)
    df.to_csv(ouput_file_path, index=False)


traslate_csv(
    file_path="/Users/akravche/Projects/UCU/alignment/evaluation/datasets/social-chem-101/social-chem-101_care-harm_rot-agree_4.csv",
    source_column="action",
    target_column="action_ukr",
    ouput_file_path="/Users/akravche/Projects/UCU/alignment/evaluation/datasets/social-chem-101/social-chem-101_care-harm_4_deepl_translated.csv",
)

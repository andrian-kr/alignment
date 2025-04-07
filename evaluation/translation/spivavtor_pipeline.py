import pandas as pd
from llm.models.spivavtor import Spivavtor
from tqdm import tqdm
from core.logger import general_logger

tqdm.pandas()


class SpivavtorPipeline:
    def __init__(self):
        self.spivavtor = Spivavtor()

    def test_pipeline(self):
        query_template = """
Зробiть текст бiльш зв’язним.
Не змінюйте відмінок. 
Не додавайте звертання.

Текст: бути поруч з другом, коли у нього проблеми."""
        x = self.spivavtor.run_inference(query_template)
        general_logger.info(f"Test query: {x}")

        query_template2 = """
Удосконалiть граматику цього тексту: бути поруч з другом, коли у нього проблеми."""
        x = self.spivavtor.run_inference(query_template2)
        general_logger.info(f"Test query: {x}")

        query_template3 = """
Перефразуйте речення: бути поруч з другом, коли у нього проблеми."""
        x = self.spivavtor.run_inference(query_template3)
        general_logger.info(f"Test query: {x}")

    def improve_coherence(
        self,
        file_path: str = "evaluation/datasets/social-chem-101/social-chem-101_care-harm_4_deepl_translated.csv",
        output_file_path: str = "evaluation/datasets/social-chem-101/social-chem-101_care-spivavtor_coherence.csv",
        input_column_name: str = "action_ukr",
        output_column_name: str = "action_ukr_spivavtor",
    ):
        df = pd.read_csv(file_path)
        query_template = """
Зробiть текст бiльш зв’язним.
Не змінюйте відмінок. 
Не додавайте звертання.

Текст: {}"""
        df[output_column_name] = df[input_column_name].progress_apply(
            lambda x: self.spivavtor.run_inference(query_template.format(x.strip()))
        )
        df.to_csv(output_file_path, index=False)

    def improve_gec(
        self,
        file_path: str,
        output_file_path: str,
        input_column_name: str,
        output_column_name: str,
    ):
        df = pd.read_csv(file_path)
        query_template = "Удосконалiть граматику цього тексту: {}"
        tqdm.pandas()
        df[output_column_name] = df[input_column_name].progress_apply(
            lambda x: self.spivavtor.run_inference(query_template.format(x.strip()))
        )
        df.to_csv(output_file_path, index=False)

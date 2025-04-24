import pandas as pd
from tqdm import tqdm

from llm.models.spivavtor import Spivavtor

tqdm.pandas()


class SpivavtorPipeline:
    def __init__(self):
        self.spivavtor = Spivavtor()

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

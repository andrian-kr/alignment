import os

from dotenv import load_dotenv
from langfuse import Langfuse
from tqdm import tqdm
import pandas as pd

load_dotenv()

langfuse_client = Langfuse()


def create_dataset(
    dataset_name: str,
    description: str,
    data: pd.DataFrame,
):
    print(f"Creating dataset {dataset_name}...")

    dataset = langfuse_client.create_dataset(
        name=dataset_name,
        description=description,
    )

    for _, row in tqdm(data.iterrows(), total=len(data)):
        langfuse_client.create_dataset_item(
            dataset_name=dataset.name,
            input={"query": row["inputs"]},
            expected_output={"targets": row["targets"]},
        )
    print(f"Dataset {dataset_name} created.")


def create_all_datasets(dataset_dir: str = "data"):
    for file in os.listdir(dataset_dir):
        if file.endswith(".csv"):
            dataset_name = file.split(".")[0]
            dataset_path = os.path.join(dataset_dir, file)
            data = pd.read_csv(dataset_path)

            create_dataset(
                dataset_name,
                f"Test inputs and target outputs for {dataset_name}.",
                data,
            )


create_all_datasets()

import os

import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse
from tqdm import tqdm

load_dotenv()

langfuse_client = Langfuse()


def create_dataset(
    dataset_name: str,
    description: str,
    data: pd.DataFrame,
    input_mapping: dict[str, str],
    output_mapping: dict[str, str],
):
    print(f"Creating dataset {dataset_name}...")

    dataset = langfuse_client.create_dataset(
        name=dataset_name,
        description=description,
    )

    data.fillna("", inplace=True)
    for _, row in tqdm(data.iterrows(), total=len(data)):
        langfuse_client.create_dataset_item(
            dataset_name=dataset.name,
            input={key: row[source_key] for key, source_key in input_mapping.items()},
            expected_output={key: row[source_key] for key, source_key in output_mapping.items()},
        )
    print(f"Dataset {dataset_name} created.")


def create_ethics_dataset(file_path: str = "data/ethics_commonsense.csv"):
    dataset_name = os.path.basename(file_path).split(".")[0]
    data = pd.read_csv(file_path)

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={"input": "input_ukr", "input_en": "input"},
        output_mapping={"label": "label"},
    )


def create_social_chem_dataset(file_path: str = "datasets/social-chem-101/social_chem_101_care_harm_4_translated.csv"):
    dataset_name = "sc_101_care_harm"
    data = pd.read_csv(file_path)

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={
            "input": "action_ukr",
            "input_en": "action",
            "area": "area",
            "rot-categorization": "rot-categorization",
        },
        output_mapping={"label": "label"},
    )


create_social_chem_dataset()


# def create_all_datasets(dataset_dir: str = "data"):
#     for file in os.listdir(dataset_dir):
#         if file.endswith(".csv"):
#             dataset_name = file.split(".")[0]
#             dataset_path = os.path.join(dataset_dir, file)
#             data = pd.read_csv(dataset_path)

#             create_dataset(
#                 dataset_name,
#                 f"Test inputs and target outputs for {dataset_name}.",
#                 data,
#             )

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langfuse import Langfuse

load_dotenv()

langfuse_client = Langfuse()


def create_dataset(
    dataset_name: str,
    description: str,
    data: pd.DataFrame,
    input_mapping: dict[str, str],
    output_mapping: dict[str, str],
    transform_functions: dict[str, callable] = None,
):
    """
    Create a dataset in Langfuse.

    Args:
        dataset_name: Name of the dataset
        description: Description of the dataset
        data: DataFrame containing the data
        input_mapping: Dict mapping Langfuse input keys to DataFrame column names
        output_mapping: Dict mapping Langfuse output keys to DataFrame column names
        transform_functions: Dict mapping DataFrame column names to transformation functions
    """
    print(f"Creating dataset {dataset_name}...")

    dataset = langfuse_client.create_dataset(
        name=dataset_name,
        description=description,
    )

    data.fillna("", inplace=True)

    transform_functions = transform_functions or {}

    for _, row in tqdm(data.iterrows(), total=len(data)):
        input_data = {}
        for key, source_key in input_mapping.items():
            value = row[source_key]
            if source_key in transform_functions:
                value = transform_functions[source_key](value)
            input_data[key] = value

        output_data = {}
        for key, source_key in output_mapping.items():
            value = row[source_key]
            if source_key in transform_functions:
                value = transform_functions[source_key](value)
            output_data[key] = value

        langfuse_client.create_dataset_item(
            dataset_name=dataset.name,
            input=input_data,
            expected_output=output_data,
        )
    print(f"Dataset {dataset_name} created.")


def create_ethics_dataset(
    file_path: str = "../data/ethics_commonsense_final.csv",
    dataset_name: str = "ethics_commonsense",
    input_column: str = "input_ukr",
):
    """Create an ethics dataset in Langfuse."""
    data = pd.read_csv(file_path)

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={"input": input_column, "input_en": "input"},
        output_mapping={"label": "label"},
    )


def create_social_chem_dataset(
    file_path: str = "../data/social-chem-101/social_chem_101_final.csv",
    dataset_name: str = "sc_101_care_harm",
    accepted_labels: list[int] = None,
    input_column: str = "action_ukr",
    input_column_en: str = "action",
):
    """Create a social chemistry dataset in Langfuse."""
    data = pd.read_csv(file_path)
    if accepted_labels is not None:
        data = data[data["label"].isin(accepted_labels)]
        print(f"Filtered dataset length: {len(data)}")

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={
            "input": input_column,
            "input_en": input_column_en,
            "area": "area",
            "rot-categorization": "rot-categorization",
        },
        output_mapping={"label": "label"},
    )


def create_mixed_pva_dataset(
    file_path: str,
    dataset_name: str,
    input_column: str = "query_ukr",
    input_column_en: str = "query_en",
):
    data = pd.read_csv(file_path)

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={"input": input_column, "input_en": input_column_en, "source": "source"},
        output_mapping={},
    )


## Create Ethics Commonsense dataset
create_ethics_dataset()

## Create Social Chemistry 101 dataset
create_social_chem_dataset()

## Create dataset for PVA evaluation
create_mixed_pva_dataset(file_path="../data/pva_full.csv", dataset_name="mixed_bad")

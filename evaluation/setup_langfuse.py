import os
import ast
import json

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


def create_ethics_dataset(file_path: str = "data/ethics_commonsense.csv", dataset_name: str = "ethics_commonsense"):
    """Create an ethics dataset in Langfuse."""
    data = pd.read_csv(file_path)

    create_dataset(
        dataset_name,
        f"Test inputs and target outputs for {dataset_name}.",
        data,
        input_mapping={"input": "input_ukr", "input_en": "input"},
        output_mapping={"label": "label"},
    )


def create_social_chem_dataset(
    file_path: str = "datasets/social-chem-101/social_chem_101_care_harm_4_translated.csv",
    dataset_name: str = "sc_101_care_harm",
):
    """Create a social chemistry dataset in Langfuse."""
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


def parse_list(value):
    """Parse a string representation of a list."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing value: {value}. Error: {e}")
            return value
    return value


def create_zno_dataset(
    file_path: str,
    dataset_name: str = None,
    subject_type: str = None,
):
    """
    Create a ZNO dataset in Langfuse.

    Args:
        file_path: Path to the ZNO dataset CSV file
        dataset_name: Name of the dataset, if None will be derived from file name
        subject_type: Subject type (history, ukr_lit, etc.), if None will be derived from file name
    """
    if dataset_name is None:
        dataset_name = os.path.basename(file_path).split(".")[0]

    if subject_type is None:
        subject_type = dataset_name.replace("zno_", "")

    print(f"Loading ZNO {subject_type} dataset from {file_path}...")
    data = pd.read_csv(file_path)

    def parse_answers(answers_str):
        answers = parse_list(answers_str)
        formatted_answers = []
        for answer in answers:
            if isinstance(answer, dict):
                formatted_answers.append(f"{answer.get('marker', '')}: {answer.get('text', '')}")
            elif isinstance(answer, str):
                formatted_answers.append(answer)
        return "\n".join(formatted_answers)

    def parse_correct_answers(correct_answers_str):
        answers = parse_list(correct_answers_str)
        return ", ".join(answers)

    create_dataset(
        dataset_name,
        f"ZNO {subject_type} test dataset",
        data,
        input_mapping={
            "question": "question",
            "answers": "answers",
        },
        output_mapping={
            "correct_answer": "correct_answers",
        },
        transform_functions={
            "answers": parse_answers,
            "correct_answers": parse_correct_answers,
        },
    )


def create_all_datasets(dataset_dir: str = None, dataset_type: str = None):
    """
    Create all datasets from the specified directory.

    Args:
        dataset_dir: Directory containing datasets
        dataset_type: Type of datasets to create (ethics, sc_101, zno)
    """
    if dataset_dir is None:
        return

    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} does not exist.")
        return

    dataset_creators = {
        "ethics": lambda file_path: create_ethics_dataset(
            file_path=file_path, dataset_name=os.path.basename(file_path).split(".")[0]
        ),
        "sc_101": lambda file_path: create_social_chem_dataset(
            file_path=file_path, dataset_name=os.path.basename(file_path).split(".")[0]
        ),
        "zno": lambda file_path: create_zno_dataset(file_path),
    }

    if dataset_type and dataset_type not in dataset_creators:
        print(f"Unknown dataset type: {dataset_type}")
        return

    for file in os.listdir(dataset_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(dataset_dir, file)

            # Determine dataset type from file or path
            detected_type = None
            if "ethics" in file or "ethics" in dataset_dir:
                detected_type = "ethics"
            elif "social-chem" in file or "sc_101" in file:
                detected_type = "sc_101"
            elif "zno" in file:
                detected_type = "zno"

            # Skip if dataset_type is specified and doesn't match
            if dataset_type and detected_type != dataset_type:
                continue

            if detected_type in dataset_creators:
                dataset_creators[detected_type](file_path)
            else:
                print(f"Unknown dataset type for file: {file}")


# Example usage - uncomment to use
# create_social_chem_dataset(
#     file_path="/Users/akravche/Projects/UCU/alignment/evaluation/datasets/social-chem-101/social-chem-101_care-harm_4_deepl_translated.csv",
#     dataset_name="sc_101_care_harm_deepl",
# )

# create_ethics_dataset(
#     file_path="/Users/akravche/Projects/UCU/alignment/evaluation/datasets/ethics/ethics_commonsense_deepl_translated.csv",
#     dataset_name="ethics_commonsense_deepl",
# )

# Upload ZNO datasets
create_zno_dataset("/Users/akravche/Projects/UCU/alignment/evaluation/datasets/zno/zno_history.csv")
create_zno_dataset("/Users/akravche/Projects/UCU/alignment/evaluation/datasets/zno/zno_ukr_lit.csv")

# To upload all datasets from a directory:
# create_all_datasets("/Users/akravche/Projects/UCU/alignment/evaluation/datasets/zno", "zno")

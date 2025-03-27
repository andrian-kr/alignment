import os

import pandas as pd
from core.logger import general_logger
from langfuse import Langfuse
from tqdm import tqdm
from dto.types import DatasetType


input_keys_mapping = {
    DatasetType.ETHICS: ["query", "input_en", "expected_output"],
    DatasetType.SOCIAL_CHEMISTRY_101: ["query", "expected_output"],
}

output_keys_mapping = {DatasetType.ETHICS: ["prediction"], DatasetType.SOCIAL_CHEMISTRY_101: ["prediction"]}


def langfuse_run_to_csv(
    client: Langfuse,
    dataset_name: str,
    run_name: str,
    output_path: str,
    input_keys: list[str],
    output_keys: list[str],
    items_count: int | None = None,
):
    """
    Export langfuse run results as a CSV file. For historical runs.
    """
    run_name = run_name.replace("/", "%2F")
    dataset_run = client.get_dataset_run(dataset_name=dataset_name, dataset_run_name=run_name)
    if items_count is not None and len(dataset_run.dataset_run_items) != items_count:
        general_logger.info(f"Skipping run {run_name} as it has only {len(dataset_run.dataset_run_items)} items")
        return

    run_items = []
    for run_item in tqdm(dataset_run.dataset_run_items):
        run_item_dict = {
            "id": run_item.id,
            "dataset_item_id": run_item.dataset_item_id,
            "trace_id": run_item.trace_id,
            "created_at": run_item.created_at,
            "updated_at": run_item.updated_at,
        }

        try:
            trace = client.get_trace(id=run_item.trace_id)
        except Exception as e:
            general_logger.error(f"Failed to fetch trace {run_item.trace_id}: {e}")
            continue

        for input_key in input_keys:
            if input_key in trace.input:
                run_item_dict[input_key] = trace.input[input_key]

        for output_key in output_keys:
            if output_key in trace.output:
                run_item_dict[output_key] = trace.output[output_key]
        run_items.append(run_item_dict)

    df = pd.DataFrame(run_items)
    df.to_csv(output_path, index=False)
    general_logger.info(f"Exported run {run_name} to {output_path}")


def langfuse_dataset_to_csv(
    client: Langfuse,
    dataset_name: str,
    dataset_type: DatasetType,
    results_dir: str,
    run_names: list[str] | None = None,
    only_full_run: bool = True,
):
    dataset_folder = os.path.join(results_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    existing_files = os.listdir(dataset_folder)
    existing_file_names = [file.removesuffix(".csv") for file in existing_files]
    general_logger.info(f"Found {len(existing_file_names)} existing files")

    input_keys = input_keys_mapping[dataset_type]
    output_keys = output_keys_mapping[dataset_type]

    items_count = None
    if only_full_run:
        dataset = client.get_dataset(name=dataset_name)
        items_count = len(dataset.items)

    if run_names is None:
        dataset_runs = client.get_dataset_runs(dataset_name=dataset_name)
        general_logger.info(f"Found {len(dataset_runs.data)} runs")
        run_names = [run.name for run in dataset_runs.data if run.name.split("/")[-1] not in existing_file_names]
        general_logger.info(f"Found {len(run_names)} new runs")

    for run in run_names:
        general_logger.info(f"Exporting run {run}")
        langfuse_run_to_csv(
            client=client,
            dataset_name=dataset_name,
            run_name=run,
            input_keys=input_keys,
            output_keys=output_keys,
            output_path=os.path.join(dataset_folder, f"{run.split("/")[-1]}.csv"),
            items_count=items_count,
        )


LANGFUSER_CLIENT = Langfuse()
DATASET_NAME = "sc_101_care_harm_deepl"
DATASET_TYPE = DatasetType.SOCIAL_CHEMISTRY_101
langfuse_dataset_to_csv(LANGFUSER_CLIENT, DATASET_NAME, DATASET_TYPE, os.getenv("RESULTS_DIR"))

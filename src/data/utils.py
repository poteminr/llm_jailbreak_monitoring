from typing import Dict, List, Union

import datasets


def process_dataset(
    data: Union[List[Dict[str, Union[str, bool]]], List[datasets.Dataset]],
    local_dataset_path: str,
    hf_dataset_path: str,
) -> None:
    """
    Process a dataset and save it to the local disk and push it to the Hugging Face Hub.

    :param data: List of dictionaries or list of datasets.
    :param local_dataset_path: Path to save the local dataset.
    :param hf_dataset_path: Path to push the dataset to Hugging Face.
    """
    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        dataset = datasets.Dataset.from_list(data)
    else:
        dataset = datasets.concatenate_datasets(data)

    dataset.save_to_disk(local_dataset_path)
    dataset.push_to_hub(repo_id=hf_dataset_path, private=True)
    return None


def remove_columns(
    dataset: datasets.Dataset, actual_columns: List[str]
) -> datasets.Dataset:
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in actual_columns]
    )
    return dataset


def rename_columns(
    dataset: datasets.Dataset,
    input_field: str | None = None,
    output_field: str | None = None,
    jailbreak_field: str | None = None,
) -> datasets.Dataset:
    columns = {}
    if input_field:
        columns[input_field] = "prompt"
    if output_field:
        columns[output_field] = "conversation"
    if jailbreak_field:
        columns[jailbreak_field] = "jailbreak"
    return dataset.rename_columns(columns)


def clear_dataset(
    data: List[Dict[str, Union[str, bool]]], fields_to_keep: List[str]
) -> List[Dict[str, Union[str, bool]]]:
    """
    Dataset cleaning: removing unnecessary fields except those passed as arguments, and checking for duplicates.

    :param data: List of dictionaries with data.
    :param fields_to_keep: List of fields to keep.
    :return: Cleaned list of dictionaries.
    """

    data = [
        {field: item[field] for field in fields_to_keep if field in item}
        for item in data
    ]
    cleaned_data = []
    seen_prompts = set()

    for item in data:
        if item["prompt"] not in seen_prompts:
            seen_prompts.add(item["prompt"])
            cleaned_data.append(item)

    return cleaned_data

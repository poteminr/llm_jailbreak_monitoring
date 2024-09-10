import json
from typing import Any, Dict

import click
import datasets
from datasets import Value

from processer import HfDataProcesser
from utils import process_dataset, remove_columns, rename_columns


@click.command()
@click.option("--output_local_dataset_path", type=click.STRING)
@click.option("--output_hf_dataset_path", type=click.STRING)
@click.option("--jailbreak_local_dataset_path", type=click.STRING)
@click.option("--jailbreak_hf_dataset_path", type=click.STRING)
def merge_data(
    output_local_dataset_path: str,
    output_hf_dataset_path: str,
    jailbreak_local_dataset_path: str,
    jailbreak_hf_dataset_path: str,
) -> None:
    """
    Merge datasets from Hugging Face and local storage.

    :param output_local_dataset_path: Path to the local output dataset.
    :param output_hf_dataset_path: Path to the output dataset on Hugging Face.
    :param jailbreak_local_dataset_path: Path to the local jailbreak dataset
    :param jailbreak_hf_dataset_path: Path to the jailbreak dataset on Hugging Face.
    """
    to_merge_output = []
    to_merge_jailbreak = []
    hf_processer = HfDataProcesser()

    with open("configs/hf_datasets.json", "r") as f:
        hf_datasets = json.load(f)

    for hf_dataset in hf_datasets["datasets"]:
        dataset_path = hf_dataset["dataset_path"]
        input_field = hf_dataset["prompt_field"]
        output_field = hf_dataset["output_field"]
        label_field = hf_dataset["label_field"]
        jailbreak_field = hf_dataset["jailbreak_field"]
        dataset_name = hf_dataset["dataset_name"]
        dataset = datasets.load_dataset(dataset_path, name=dataset_name)

        for split in dataset.keys():
            split_dataset = dataset[split]

            if label_field in split_dataset.column_names:
                # Обработка классов токсичности
                pass

            if jailbreak_field in split_dataset.column_names:
                # Обработка jailbreak запросов

                def cast_to_int(item: Dict[str, Any]) -> Dict[str, Any]:
                    item[jailbreak_field] = hf_processer.transform_jailbreak_field(
                        dataset_path, item[jailbreak_field]
                    )
                    return item

                jailbreak_dataset = split_dataset.filter(
                    lambda item: bool(
                        hf_processer.transform_jailbreak_field(
                            dataset_path, item[jailbreak_field]
                        )
                    )
                )
                jailbreak_dataset = jailbreak_dataset.map(
                    lambda item: cast_to_int(item)
                )
                jailbreak_dataset = remove_columns(
                    jailbreak_dataset, [input_field, jailbreak_field]
                )
                jailbreak_dataset = rename_columns(
                    jailbreak_dataset,
                    input_field=input_field,
                    jailbreak_field=jailbreak_field,
                )

                new_features = jailbreak_dataset.features.copy()
                new_features["jailbreak"] = Value("int64")
                jailbreak_dataset = jailbreak_dataset.cast(new_features)
                to_merge_jailbreak.append(jailbreak_dataset)

            if output_field in split_dataset.column_names:
                # Обработка генерации модели
                split_dataset = remove_columns(
                    split_dataset, [input_field, output_field]
                )
                split_dataset = rename_columns(
                    split_dataset, input_field=input_field, output_field=output_field
                )
                to_merge_output.append(split_dataset)

    if to_merge_jailbreak:
        initial_jaiblreak_dataset = datasets.load_dataset(jailbreak_hf_dataset_path)
        to_merge_jailbreak.append(initial_jaiblreak_dataset["train"])

    process_dataset(to_merge_output, output_local_dataset_path, output_hf_dataset_path)
    process_dataset(
        to_merge_jailbreak, jailbreak_local_dataset_path, jailbreak_hf_dataset_path
    )

    return None


if __name__ == "__main__":
    merge_data()

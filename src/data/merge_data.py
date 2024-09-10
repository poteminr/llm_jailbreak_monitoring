import json

import click
import datasets

from data.processer import HfDataProcesser
from data.utils import process_dataset, remove_columns, rename_columns


@click.command()
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--output_hf_dataset_path", type=click.STRING)
@click.option("--jailbreak_hf_dataset_path", type=click.STRING)
def merge_data(
    local_dataset_path: str, output_hf_dataset_path: str, jailbreak_hf_dataset_path: str
) -> None:
    """
    Merge datasets from Hugging Face and local storage.

    :param local_dataset_path: Path to the local dataset.
    :param output_hf_dataset_path: Path to the output dataset on Hugging Face.
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
        dataset = datasets.load_dataset(dataset_path)

        for split in dataset.keys():
            split_dataset = dataset[split]

            if label_field in split_dataset.column_names:
                # Обработка классов токсичности
                pass

            if jailbreak_field in split_dataset.column_names:
                # Обработка jailbreak запросов
                jailbreak_dataset = split_dataset.filter(
                    lambda item: hf_processer.is_jailbreak(
                        dataset_path, item[jailbreak_field]
                    )
                )
                jailbreak_dataset = remove_columns(
                    jailbreak_dataset, [input_field, jailbreak_field]
                )
                jailbreak_dataset = rename_columns(
                    jailbreak_dataset,
                    input_field=input_field,
                    jailbreak_field=jailbreak_field,
                )
                to_merge_jailbreak.extend(jailbreak_dataset)

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
        to_merge_jailbreak.extend(initial_jaiblreak_dataset)

    process_dataset(to_merge_output, local_dataset_path, output_hf_dataset_path)
    process_dataset(to_merge_jailbreak, local_dataset_path, jailbreak_hf_dataset_path)

    return None


if __name__ == "__main__":
    merge_data()

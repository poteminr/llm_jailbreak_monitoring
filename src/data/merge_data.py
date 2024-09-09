import json

import click
import datasets


@click.command()
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--hf_dataset_path", type=click.STRING)
def merge_data(local_dataset_path: str, hf_dataset_path: str) -> None:
    to_merge_output = []
    to_merge_jailbreak = []

    with open("configs/hf_datasets.json", "r") as f:
        hf_datasets = json.load(f)

    for hf_dataset in hf_datasets["datasets"]:
        dataset = datasets.load_dataset(hf_dataset["dataset_path"])
        splits = dataset.keys()

        input_field = hf_dataset["prompt_field"]
        output_field = hf_dataset["output_field"]
        label_field = hf_dataset["label_field"]
        jailbreak_field = hf_dataset["jailbreak_field"]

        for split in splits:
            split_dataset = dataset[split]

            if label_field in split_dataset.column_names:
                # Обработка классов токсичности
                pass

            if jailbreak_field in split_dataset.column_names:
                # Обработка промптов, которые считаются jailbreak'ом
                pass

            split_dataset = split_dataset.remove_columns(
                [
                    col
                    for col in split_dataset.column_names
                    if col not in [input_field, output_field]
                ]
            )

            split_dataset = split_dataset.rename_columns(
                {input_field: "prompt", output_field: "conversation"}
            )

            to_merge_output.append(split_dataset)

    if to_merge_output:
        merged_dataset = datasets.concatenate_datasets(to_merge_output)
        merged_dataset.save_to_disk(local_dataset_path)
        merged_dataset.push_to_hub(repo_id=hf_dataset_path, private=True)

    if to_merge_jailbreak:
        # Объединение jailbreak датасета
        pass

    return None


if __name__ == "__main__":
    merge_data()

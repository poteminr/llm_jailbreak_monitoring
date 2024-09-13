import os

import click
import datasets
from dotenv import load_dotenv
import requests
import tqdm

from utils import process_dataset


TRANSLATE_API_URI = "https://translate.api.cloud.yandex.net/translate/v2/translate"


@click.command()
@click.option("--input_dataset_path", type=click.STRING)
@click.option("--output_local_dataset_path")
@click.option("--output_hf_dataset_path", type=click.STRING)
def translate_dataset(
    input_dataset_path: str, output_local_dataset_path: str, output_hf_dataset_path: str
) -> None:
    IAM_TOKEN = os.getenv("IAM_TOKEN")
    folder_id = os.getenv("FOLDER_ID")
    target_language = "ru"

    dataset = datasets.load_dataset(input_dataset_path, split="train")
    translated_dset = []
    langs = []

    text_field = "prompt" if "prompt" in dataset.column_names else "conversation"
    label_field = "jailbreak" if text_field == "prompt" else "toxic"

    batch_size = 5
    dataset_len = len(dataset)
    for i in tqdm.tqdm(range(0, dataset_len, batch_size)):
        chunk = dataset.select(range(i, min(dataset_len, i + batch_size)))
        texts = chunk[text_field]
        labels = chunk[label_field]

        body = {
            "targetLanguageCode": target_language,
            "texts": texts,
            "folderId": folder_id,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(IAM_TOKEN),
        }

        response = requests.post(
            TRANSLATE_API_URI,
            json=body,
            headers=headers,
        )

        print(response.json())
        translations = response.json()["translations"]

        langs.extend(
            [translation["detectedLanguageCode"] for translation in translations]
        )

        translated_dset.extend(
            [
                {
                    text_field: translation["text"],
                    label_field: label,
                    "language": "ru",
                }
                for translation, label in zip(translations, labels)
            ]
        )

    dataset = dataset.add_column("language", langs)
    translated_dataset = datasets.Dataset.from_dict(translated_dset)

    process_dataset(
        [translated_dataset, dataset], output_local_dataset_path, output_hf_dataset_path
    )
    return None


if __name__ == "__main__":
    load_dotenv()
    translate_dataset()

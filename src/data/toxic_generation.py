import os

import click
import datasets
from dotenv import load_dotenv
import requests


class LlmApi:
    """
    Class for working with llama.cpp API
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def get_completion(self, prompt: str) -> str:
        response = requests.post(self.uri, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()


@click.command()
@click.option("--prompt_dataset_path", type=click.STRING)
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--hf_dataset_path", type=click.STRING)
def create_answers(
    prompt_dataset_path: str, local_dataset_path: str, hf_dataset_path: str
) -> None:
    """
    Generate answers for prompts from a dataset and save the results.

    :param prompt_dataset_path: Path to the dataset containing prompts.
    :param local_dataset_path: Path to save the local dataset.
    :param hf_dataset_path: Path to push the dataset to Hugging Face.
    """
    prompt_dataset = datasets.load_dataset(prompt_dataset_path)
    llm_api = LlmApi(os.getenv("URI"), os.getenv("SECRET_WORD"))
    print("Dataset downloaded")

    for prompt in prompt_dataset["train"]["prompt"]:
        res = llm_api.get_completion(prompt)
        print(res)
        break
    return None


if __name__ == "__main__":
    load_dotenv()
    create_answers()

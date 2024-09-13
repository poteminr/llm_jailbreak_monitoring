import os
import random

import click
import datasets
from dotenv import load_dotenv
import requests
import tqdm

from src.utils import process_dataset


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
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--hf_dataset_path", type=click.STRING)
@click.opion("--num_samples", type=click.INT)
def create_answers(
    local_dataset_path: str, hf_dataset_path: str, num_samples: int
) -> None:
    """
    Generate answers for prompts from a dataset and save the results.

    :param prompt_dataset_path: Path to the dataset containing prompts.
    :param local_dataset_path: Path to save the local dataset.
    :param hf_dataset_path: Path to push the dataset to Hugging Face.
    """
    generated_data = []
    prompt_dataset = datasets.load_dataset(
        "lmsys/toxic-chat", "toxicchat0124", split="train"
    )
    llm_api = LlmApi(os.getenv("URI"))
    toxic_prompts = prompt_dataset.filter(lambda item: item["toxicity"])["user_input"]
    if len(toxic_prompts) < num_samples:
        num_samples = len(toxic_prompts)
    rand = random.choices(toxic_prompts, k=num_samples)
    for prompt in tqdm.tqdm(rand):
        res = llm_api.get_completion(prompt)
        generated_data.append({"prompt": prompt, "conversation": res})

    process_dataset(generated_data, local_dataset_path, hf_dataset_path)
    return None


if __name__ == "__main__":
    load_dotenv()
    create_answers()

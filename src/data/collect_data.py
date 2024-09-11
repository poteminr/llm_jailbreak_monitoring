import json
import os
from typing import Dict, List, Union

import click
from dotenv import load_dotenv
from github import Auth, Github

from utils import clear_dataset, process_dataset
from processer import GitDataProcesser, NotFoundDataError


def collect_recursively_from_github(
    g: Github, repo_path: str, initial_path: str, data_processer: GitDataProcesser
) -> List[Dict[str, Union[str, bool]]]:
    repo = g.get_repo(repo_path)
    contents = repo.get_contents(initial_path)
    data = []

    def is_valid_extension(download_url: str) -> bool:
        return download_url.split(".")[-1] in {"csv", "json"}

    while contents:
        content_file = contents.pop(0)
        if content_file.type == "dir":
            contents.extend(repo.get_contents(content_file.path))
        elif is_valid_extension(content_file.download_url):
            try:
                print(f"Processing {content_file.download_url}")
                data.extend(data_processer.process_filedata(repo_path, content_file))
            except NotFoundDataError:
                print(f"Data not found for {content_file.download_url}")

    return data


@click.command()
@click.option("--output_path", type=click.STRING)
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--hf_dataset_path", type=click.STRING)
def collect(output_path: str, local_dataset_path: str, hf_dataset_path: str) -> None:
    """
    Collect data from GitHub repositories and save it to a local dataset and push it to Hugging Face.

    :param output_path: Path to save the output dataset.
    :param local_dataset_path: Path to save the local dataset.
    :param hf_dataset_path: Path to push the dataset to Hugging Face.
    """
    all_data = []
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    g = Github(auth=auth)
    data_processer = GitDataProcesser()

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_data = json.load(f)["data"]
    else:
        with open("configs/git_repos.json", "r") as f:
            git_repos = json.load(f)

        for git_repo in git_repos["repos"]:
            data_chunk = collect_recursively_from_github(
                g, git_repo["repo_path"], git_repo["initial_path"], data_processer
            )
            all_data.extend([item for item in data_chunk if item])

    all_data = clear_dataset(
        all_data,
        [
            data_processer.input_field,
            data_processer.target_field,
            data_processer.source_field,
        ],
    )

    with open(output_path, "w") as f:
        json.dump({"data": all_data}, f)

    process_dataset(all_data, local_dataset_path, hf_dataset_path)
    return None


if __name__ == "__main__":
    load_dotenv()
    collect()

import json
import os
from typing import Dict, List, Union

import click
from datasets import Dataset
from dotenv import load_dotenv
from github import Auth, Github

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


def create_huggingface_dataset(
    data: List[Dict[str, Union[str, bool]]],
    local_dataset_path: str,
    hf_dataset_path: str,
) -> None:
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(local_dataset_path)
    dataset.push_to_hub(repo_id=hf_dataset_path, private=True)
    return None


@click.command()
@click.option("--output_path", type=click.STRING)
@click.option("--local_dataset_path", type=click.STRING)
@click.option("--hf_dataset_path", type=click.STRING)
def collect(output_path: str, local_dataset_path: str, hf_dataset_path: str) -> None:
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

            def verify_item(item: Dict[str, Union[str, bool]]) -> bool:
                return bool(item.get("prompt")) and item["jailbreak"]

            all_data.extend([item for item in data_chunk if verify_item(item)])

        with open(output_path, "w") as f:
            json.dump({"data": all_data}, f)

    create_huggingface_dataset(all_data, local_dataset_path, hf_dataset_path)
    return None


if __name__ == "__main__":
    load_dotenv()
    collect()

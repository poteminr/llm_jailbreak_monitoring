import json
from typing import Any, Dict, List, Union
import urllib

from github import ContentFile
import numpy as np
import pandas as pd
import requests


class NotFoundDataError(Exception):
    pass


class GitDataProcesser:
    def __init__(self) -> None:
        self.verazuo = "verazuo/jailbreak_llms"
        self.jailbreakbench = "JailbreakBench/artifacts"
        self.sherdencooper = "sherdencooper/GPTFuzz"
        self.tml_epfl = "tml-epfl/llm-adaptive-attacks"

        self.input_field = "prompt"
        self.target_field = "jailbreak"

        self.verazuo_fields = ["prompt", "jailbreak"]
        self.jailbreakbench_fields = ["prompt", "jailbroken"]
        self.sherdencooper_field = "text"

    def _encode_field(self, key: str) -> str:
        if key == "jailbroken":
            return self.target_field
        return key

    def _encode_value(self, value: Union[str, bool]) -> Union[str, int]:
        if isinstance(value, bool):
            return int(value)
        return value

    def _process_sherdencooper_filedata(
        self, content_file: ContentFile
    ) -> List[Dict[str, Union[str, bool]]]:
        df = pd.read_csv(content_file.download_url)
        df[self.input_field] = df[self.sherdencooper_field]
        df[self.target_field] = np.ones(df.shape[0], dtype=np.int64)
        df = df.drop(self.sherdencooper_field, axis=1)
        if "id" in df.columns:
            df = df.drop(["id"], axis=1)
        return df.to_dict(orient="records")

    def _process_verazuo_filedata(
        self, content_file: ContentFile
    ) -> List[Dict[str, Union[str, bool]]]:
        try:
            df = pd.read_csv(content_file.download_url)
        except urllib.error.URLError:
            response = requests.get(content_file.download_url)
            if response.status_code == 200:
                with open("local_file.csv", "wb") as f:
                    f.write(response.content)
                df = pd.read_csv("local_file.csv")
            else:
                print(f"Failed to download the file: {response.status_code}")
        df = df[self.verazuo_fields]
        df[self.target_field] = df[self.target_field].astype(int)
        return df.to_dict(orient="records")

    def _process_jailbreakbench_filedata(
        self, content_file: ContentFile
    ) -> List[Dict[str, Union[str, bool]]]:
        data = json.loads(content_file.decoded_content)
        jailbreak_data = data.get("jailbreaks", None)
        if jailbreak_data:
            data = [
                {
                    self._encode_field(k): self._encode_value(v)
                    for k, v in jailbreak_item.items()
                    if k in self.jailbreakbench_fields
                }
                for jailbreak_item in jailbreak_data
            ]
            return data
        else:
            raise NotFoundDataError

    def process_filedata(
        self, source_repo: str, content_file: ContentFile
    ) -> List[Dict[str, Union[bool, str]]]:
        """
        Process the data from the given content file.

        :param source_repo: Path to the repository containing the data.
        :param content_file: Content file to process.
        :return: List of processed data.
        """
        match source_repo:
            case self.verazuo:
                return self._process_verazuo_filedata(content_file)
            case self.jailbreakbench:
                return self._process_jailbreakbench_filedata(content_file)
            case self.sherdencooper:
                return self._process_sherdencooper_filedata(content_file)
            case self.tml_epfl:
                return self._process_jailbreakbench_filedata(content_file)


class HfDataProcesser:
    def __init__(self) -> None:
        self.lmsys = "lmsys/toxic-chat"
        self.rubend18 = "rubend18/ChatGPT-Jailbreak-Prompts"
        self.hackaprompt = "hackaprompt/hackaprompt-dataset"

    def transform_jailbreak_field(self, dataset_path: str, value: Any) -> int:
        """
        Check if the value is a jailbreak.

        :param dataset_path: Path to the dataset.
        :param value: Value to check.
        :return: True if the value is a jailbreak, False otherwise.
        """

        match dataset_path:
            case self.lmsys:
                return int(value)
            case self.rubend18:
                return int(value > 0)
            case self.hackaprompt:
                return int(value)
            case _:
                raise ValueError(f"Unknown dataset path: {dataset_path}")

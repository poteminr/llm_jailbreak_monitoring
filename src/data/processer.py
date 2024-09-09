import json
from typing import Dict, List, Union
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

    def _process_sherdencooper_filedata(
        self, content_file: ContentFile
    ) -> List[Dict[str, Union[str, bool]]]:
        df = pd.read_csv(content_file.download_url)
        df[self.input_field] = df[self.sherdencooper_field]
        df[self.target_field] = np.ones(df.shape[0]).astype(bool)
        df = df.drop(self.sherdencooper_field, axis=1)
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
        return df.to_dict(orient="records")

    def _process_jailbreakbench_filedata(
        self, content_file: ContentFile
    ) -> List[Dict[str, Union[str, bool]]]:
        data = json.loads(content_file.decoded_content)
        jailbreak_data = data.get("jailbreaks", None)
        if jailbreak_data:
            data = [
                {
                    self._encode_field(k): v
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
        match source_repo:
            case self.verazuo:
                return self._process_verazuo_filedata(content_file)
            case self.jailbreakbench:
                return self._process_jailbreakbench_filedata(content_file)
            case self.sherdencooper:
                return self._process_sherdencooper_filedata(content_file)
            case self.tml_epfl:
                return self._process_jailbreakbench_filedata(content_file)

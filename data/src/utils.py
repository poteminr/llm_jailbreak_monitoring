from typing import Dict, List, Union

import datasets
from datasketch import MinHash, MinHashLSH
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def process_dataset(
    data: Union[List[Dict[str, Union[str, bool]]], List[datasets.Dataset]],
    local_dataset_path: str,
    hf_dataset_path: str,
) -> None:
    """
    Process a dataset and save it to the local disk and push it to the Hugging Face Hub.

    :param data: List of dictionaries or list of datasets.
    :param local_dataset_path: Path to save the local dataset.
    :param hf_dataset_path: Path to push the dataset to Hugging Face.
    """
    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        dataset = datasets.Dataset.from_list(data)
    else:
        dataset = datasets.concatenate_datasets(data)

    dataset.save_to_disk(local_dataset_path)
    dataset.push_to_hub(repo_id=hf_dataset_path, private=True)
    return None


def remove_columns(
    dataset: datasets.Dataset, actual_columns: List[str]
) -> datasets.Dataset:
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in actual_columns]
    )
    return dataset


def rename_columns(
    dataset: datasets.Dataset,
    input_field: str | None = None,
    output_field: str | None = None,
    jailbreak_field: str | None = None,
) -> datasets.Dataset:
    columns = {}
    if input_field:
        columns[input_field] = "prompt"
    if output_field:
        columns[output_field] = "conversation"
    if jailbreak_field:
        columns[jailbreak_field] = "jailbreak"
    return dataset.rename_columns(columns)


def clear_dataset(
    data: List[Dict[str, Union[str, int]]], fields_to_keep: List[str]
) -> List[Dict[str, Union[str, int]]]:
    """
    Dataset cleaning: removing unnecessary fields except those passed as arguments, and checking for duplicates.

    :param data: List of dictionaries with data.
    :param fields_to_keep: List of fields to keep.
    :return: Cleaned list of dictionaries.
    """

    data = [
        {field: item[field] for field in fields_to_keep if field in item}
        for item in data
    ]
    cleaned_data = []
    seen_prompts = set()

    for item in data:
        if (item["prompt"]) and (item["prompt"] not in seen_prompts):
            seen_prompts.add(item["prompt"])
            cleaned_data.append(item)

    return cleaned_data


def remove_duplicates(
    raw_dataset: datasets.Dataset, threshold: float = 0.6, num_perm: int = 256
) -> datasets.Dataset:
    vectorizer = CountVectorizer()
    prompts = raw_dataset["prompt"]
    X = vectorizer.fit_transform(prompts)
    # text_vectors = X.toarray()

    def get_minhash_signature(vector: np.ndarray) -> MinHash:
        minhash = MinHash(num_perm=num_perm)
        for idx in vector.nonzero()[1]:  # проходим по индексам ненулевых значений
            minhash.update(str(idx).encode("utf8"))
        return minhash

    minhash_signatures = [get_minhash_signature(vec) for vec in X]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for i, minhash in enumerate(minhash_signatures):
        lsh.insert(f"doc_{i}", minhash)

    duplicates = set()  # Храним индексы дубликатов

    # Храним уже обработанные группы, чтобы не обрабатывать их несколько раз
    processed_groups = set()

    for i, minhash in enumerate(minhash_signatures):
        result = lsh.query(minhash)

        # Если результат содержит больше одной строки и группа ещё не обработана
        if len(result) > 1 and tuple(sorted(result)) not in processed_groups:
            processed_groups.add(
                tuple(sorted(result))
            )  # Добавляем группу в обработанные

            # Оставляем первую строку, остальные помечаем как дубликаты
            for doc in result[1:]:  # Начиная со второй строки
                duplicates.add(int(doc.split("_")[1]))

    duplicate_indices = list(duplicates)

    # 7. Удаление строк-дубликатов из датасета Huggingface
    # Создаём новый датасет без строк, которые помечены как дубликаты
    dataset_filtered = raw_dataset.select(
        [i for i in range(len(dataset)) if i not in duplicate_indices]
    )

    # Выводим количество удалённых дубликатов и уникальных строк
    print(f"Total texts: {len(prompts)}")
    print(f"Removed duplicates: {len(duplicates)}")
    print(f"Unique string: {len(dataset_filtered)}")

    return dataset_filtered


if __name__ == "__main__":
    dataset = datasets.load_dataset("To-the-moon/jailbreak_prompts_v2", split="train")

    filtered_dataset = remove_duplicates(dataset)
    print(f"Jailbreak prompts: {sum(filtered_dataset['jailbreak'])}")
    print(f"All prompts: {len(filtered_dataset['prompt'])}")

    filtered_dataset.push_to_hub(
        repo_id="To-the-moon/jailbreak_prompts_v3", private=True
    )

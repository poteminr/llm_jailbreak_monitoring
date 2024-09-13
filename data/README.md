# Datasets

Скрипты для создания датасетов

## Ключи доступа

Для правильной работы необходимо получить 2 `access_token`'a:
- https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
- https://huggingface.co/docs/hub/security-tokens

Гитхабовский должен быть добавлен в переменную окружения, а токен с huggingface используется для аутентификации через `huggingface-cli`

## Создание jailbreak датасета из источнигов на github

Пример команды:
```bash
python3 data/collect_data.py --output_path data.json --local_dataset_path jailbreak_prompts --hf_dataset_path To-the-moon/jailbreak_prompts
```

Скрипт проходит по репозиториям, указанным в `configs/git_repos.json` и загружает оттуда данные о попытках jailbreak'а языковых моделей. В результате остаются 2 поля: `prompt` и `jailbreak` - показывающее прошла данная атака или нет.

Также параллельно с этим датасет заливается на huggingface.

## Дополнение датасета источниками с huggingface


Пример команды:
```bash
python src/data/merge_data.py --output_local_dataset_path ./data/toxic_output_dpo --output_hf_dataset_path To-the-moon/toxic_output --jailbreak_local_dataset_path ./data/jailbreak_prompts_merged --jailbreak_hf_dataset_path To-the-moon/jailbreak_prompts_v2
```

Скрипт проходит по датасетам, указанным в `configs/hf_datasets.json` и загружает оттуда данные о токсичных генерациях и jailbreak промптах. В конце создается новый датасет с выходом модели и объединяется датасет jaiblreak промптов.
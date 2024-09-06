# Datasets

Скрипты для создания датасетов

## Ключи доступа

Для правильной работы необходимо получить 2 `access_token`'a:
- https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
- https://huggingface.co/docs/hub/security-tokens

Гитхабовский должен быть добавлен в переменную окружения, а токен с huggingface используется для аутентификации через `huggingface-cli`

## Создание jailbreak датасета

Пример команды:
```bash
python3 data/collect_data.py --output_path data.json --local_dataset_path jailbreak_prompts --hf_dataset_path To-the-moon/jailbreak_prompts
```

Скрипт проходит по репозиториям, указанным в `configs/git_repos.json` и загружает оттуда данные о попытках jailbreak'а языковых моделей. В результате остаются 2 поля: `prompt` и `jailbreak` - показывающее прошла данная атака или нет.

Также параллельно с этим датасет заливается на huggingface.
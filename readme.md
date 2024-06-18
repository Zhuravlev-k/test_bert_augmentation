# Тестовое задание

## Описание задания

Реализовать базовый Embedding MixUp метод для аугментации данных для файнтюнинга bert-base-cased на датасете [rotten tomatoes](https://huggingface.co/datasets/rotten_tomatoes). 
___
## Структура проекта
- `baseline.ipynb/baseline.pdf` - Базовый пайплайн с графиками, выводами и прочим. Настоятельно рекомендуется сначала посмотреть туда
- `main.py` - Главный скрипт/точка входа для файнтюнинга и валидации 
- `train.py` - Модуль для дообучения 
- `inference.py` - Модуль для валидации 
- `model.py` - Модуль с кастомным классом тренера
- `utils.py` - Модуль с вспомогательными функциями
- `config.py` - Модуль-конфиг для хранения констант и параметров тренировки
- `requirements.txt` - Файл с зависимостями 
___
## Установка

1. Клонируйте репозиторий с GitHub и перейдите в него:

    ```sh
    git clone https://github.com/Zhuravlev-k/test_bert_augmentation.git
    ```

2. Создайте и активируйте виртуальное окружение:
3. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```
___
## Как использовать
### Тренировка модели
Выставьте необходимые параметры в ```config.py```, а затем запустите
```sh
python main.py --mode train
```
### Валидация
Убедитесь, что вы либо обучили модель и веса сохранились в папку model_mixup, либо вы загрузили параметры модели [отсюда](https://disk.yandex.ru/d/_my5c8O5ogPRQg)
```sh
python main.py --mode inference --model_dir {папка с парметрами} --dataset {validation/test}
```

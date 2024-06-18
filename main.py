"""
Главный модуль для файнтюнинга и валидации 
"""

import argparse
from train import train_model
from inference import inference_model

def main():
    """
    Главная функция для распарса аргументов и запуска файнтюна/валидации
    """
    parser = argparse.ArgumentParser(description="BERT with Mixup Training")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True,
                         help="Mode: тренируем или инференс")
    parser.add_argument("--model_dir", type=str, default="model_mixup", 
                        help="Директория для загрузки весов для валидации")
    parser.add_argument("--dataset", type=str, choices=["validation", "test"], 
                        default="validation", help="Выбор датасета для валидации")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "inference":
        inference_model(args.model_dir, args.dataset)

if __name__ == "__main__":
    main()

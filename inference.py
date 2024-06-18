"""
Модуль для валидации 
    
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from model import MixupTrainer
from utils import tokenize_function, compute_metrics
from config import training_args, num_labels, alpha, beta
import argparse

def inference_model(model_dir, dataset_type, alpha=alpha, beta=beta):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if dataset_type == "test":
        eval_dataset = tokenized_datasets["test"]
    else:
        eval_dataset = tokenized_datasets["validation"]

    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)

    trainer = MixupTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        alpha=alpha,
        beta=beta
    )

    results = trainer.evaluate()
    print(results)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Директория для загрузки весов для валидации")
    parser.add_argument("--dataset", type=str, choices=["validation", "test"], required=True, help="Выбор датасета для валидации")
    args = parser.parse_args()
    inference_model(args.model_dir, args.dataset)
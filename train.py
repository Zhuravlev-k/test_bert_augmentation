"""
Скрипт для обучения берта
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from utils import tokenize_function, compute_metrics, SaveMetricsCallback
from model import MixupTrainer
from config import training_args, num_labels, alpha, beta

def train_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.to(device)

    trainer = MixupTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        alpha=alpha,
        beta=beta
    )

    train_metrics, eval_metrics = [], [] # пусть остаётся, если что возврщать будем их
    trainer.add_callback(SaveMetricsCallback(train_metrics, eval_metrics))
    trainer.train()

    # схороняем модель
    trainer.save_model(training_args.output_dir)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_model()

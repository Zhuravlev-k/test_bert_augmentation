"""
Модуль-конфиг. Здесь хранятся парпаметры тренировки и константы    
"""

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model_mixup", # ДИРЕКТОРИЮ МЕНЯТЬ ТУТ
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=8e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1, # ЭПОХИ МЕНЯТЬ ТУТ
    save_total_limit=1,  # количество сохраняемых моделей
    load_best_model_at_end=True,  
    metric_for_best_model="accuracy", 
)


num_labels = 2
alpha = 1.0
beta = 1.0
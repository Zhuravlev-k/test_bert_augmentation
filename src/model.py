"""
Модуль с кастомным классом тренера 
"""

import numpy as np
import torch
from transformers import Trainer
import torch.nn.functional as F

class MixupTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha # наши основные гиперпараметры которыми мы влияем на распределние
        # при помощи них мы можем регулировать вероятность соотношения классов
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids") 
        labels = inputs.get("labels")        
        attention_mask = inputs.get("attention_mask")

        if self.alpha > 0 and self.beta:  # при alpha/beta < 0 бета распределение не существует
            # Mixup
            lam = np.random.beta(self.alpha, self.beta)  # то самое соотношение классов
            batch_size = input_ids.size(0)
            index = torch.randperm(batch_size).to(input_ids.device) # перемещиваем индексы батча
            labels_a, labels_b = labels, labels[index]

            # получаем эмбеддинги
            embeddings = model.bert.embeddings(input_ids)
            embeddings_b = model.bert.embeddings(input_ids[index])

            # общий вид mixed = lam * a + (1 - lam) * b
            mixed_embeddings = lam * embeddings + (1 - lam) * embeddings_b
            mixed_attention_mask = attention_mask  # attention_mask не меняем

            # обновляем inputs 
            mixed_inputs = {
                "inputs_embeds": mixed_embeddings,
                "attention_mask": mixed_attention_mask,
                "labels": labels
            }
            # forward
            outputs = model(**mixed_inputs)
            logits = outputs.get("logits")
            # loss для Mixup
            loss = lam * F.cross_entropy(logits, labels_a) + (1 - lam) * F.cross_entropy(logits, labels_b)
        else: # если распределения не сущ-ет, то просто тюнем модель
            outputs = model(**inputs)
            logits = outputs("logits")
            loss = F.cross_entropy()(logits, labels)

        return (loss, outputs) if return_outputs else loss
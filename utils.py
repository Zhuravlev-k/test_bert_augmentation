"""
Модуль с вспомогательными функциями

Классы:
     - SaveMetricsCallback: класс для сохранения метрик во время обучения. Передаётся в train
     
Функции:
    - tokenize_function: функция для токенизирования датасета. Передаётся в train
    - compute_metrics: функция для рассчёта метрик. Передаётся в train
   
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer
from transformers.trainer_callback import TrainerCallback

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) # вытаксивааем наиболее вероятных класс
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class SaveMetricsCallback(TrainerCallback):
    def __init__(self, train_metrics, eval_metrics):
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics

    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.train_metrics.append(state.log_history[-1])
            if 'eval_loss' in state.log_history[-1]:
                self.eval_metrics.append(state.log_history[-1])